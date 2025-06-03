import sys
import os
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import *
from tensorflow.keras import layers, regularizers
from Observation import observation
from Markov_state_transition import Markov_state_transition_matrix, state_evolution
from matplotlib import rcParams
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLabel, QLineEdit, QPushButton, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QDoubleValidator, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

class ActorWithAttention(tf.keras.Model):
    def __init__(self, state_size, action_size, batch_norm=True,
                 hidden=[256, 256, 256], num_heads=4, key_dim=4, num_groups=16):
        super(ActorWithAttention, self).__init__()

        # 基础网络层
        self.fc1 = layers.Dense(hidden[0], input_shape=(None, state_size),
                                kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(action_size, kernel_regularizer=regularizers.l2(0.001))

        # 注意力机制
        self.state_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=hidden[0] // num_groups  # 新增value_dim对齐
        )

        # 分组参数验证
        assert hidden[0] % num_groups == 0, "hidden[0]必须能被num_groups整除"
        self.num_groups = num_groups
        self.group_dim = hidden[0] // num_groups
        self.hidden = hidden

        # 其他配置保持不变
        self.bn_layers = [layers.BatchNormalization() for _ in range(3)]
        self.batch_norm = batch_norm

    def call(self, inputs):
        x = inputs

        # 第一全连接层
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_layers[0](x)
        x = tf.nn.relu(x)

        # === 状态注意力模块 ===
        # 分组处理
        batch_size = tf.shape(x)[0]
        seq_x = tf.reshape(x, (batch_size, self.num_groups, self.group_dim))

        # 注意力计算
        attn_output = self.state_attention(seq_x, seq_x)

        # 维度恢复
        x = tf.reshape(attn_output, (batch_size, self.hidden[0]))

        # 残差块（保持不变）
        res1 = x
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn_layers[1](x)
        x = tf.nn.relu(x)
        x += res1

        res2 = x
        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn_layers[2](x)
        x = tf.nn.relu(x)
        x += res2
        return self.fc4(x)

class CriticWithAttention(tf.keras.Model):
    def __init__(self, state_size, batch_norm=True,
                 hidden=[256, 256, 256], num_heads=4, key_dim=4, num_groups=16):
        super(CriticWithAttention, self).__init__()

        # 基础网络层
        self.fc1 = layers.Dense(hidden[0], input_shape=(None, state_size),
                                kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(1, kernel_regularizer=regularizers.l2(0.001))

        # 注意力机制
        self.state_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=hidden[0] // num_groups  # 新增value_dim对齐
        )

        # 分组参数验证
        assert hidden[0] % num_groups == 0, "hidden[0]必须能被num_groups整除"
        self.num_groups = num_groups
        self.group_dim = hidden[0] // num_groups
        self.hidden = hidden

        # 其他配置保持不变
        self.bn_layers = [layers.BatchNormalization() for _ in range(3)]
        self.batch_norm = batch_norm

    def call(self, inputs):
        x = inputs

        # 第一全连接层
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_layers[0](x)
        x = tf.nn.relu(x)

        # === 状态注意力模块 ===
        # 分组处理
        batch_size = tf.shape(x)[0]
        seq_x = tf.reshape(x, (batch_size, self.num_groups, self.group_dim))

        # 注意力计算
        attn_output = self.state_attention(seq_x, seq_x)

        # 维度恢复
        x = tf.reshape(attn_output, (batch_size, self.hidden[0]))

        # 残差块（保持不变）
        res1 = x
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn_layers[1](x)
        x = tf.nn.relu(x)
        x += res1

        res2 = x
        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn_layers[2](x)
        x = tf.nn.relu(x)
        x += res2

        return self.fc4(x)

class EditableTableWidget(QTableWidget):
    """支持复制粘贴的自定义表格组件"""

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            self.paste_data()
        else:
            super().keyPressEvent(event)

    def paste_data(self):
        clipboard = QApplication.clipboard()
        data = clipboard.text()
        rows = data.split('\n')
        current_row = self.currentRow()
        current_col = self.currentColumn()

        for i, row in enumerate(rows):
            if not row.strip():
                continue
            cells = row.split('\t')
            target_row = current_row + i
            if target_row >= self.rowCount():
                self.insertRow(target_row)

            for j, cell in enumerate(cells):
                target_col = current_col + j
                if target_col >= self.columnCount():
                    break
                if target_row < self.rowCount() and target_col < self.columnCount():
                    self.setItem(target_row, target_col, QTableWidgetItem(cell.strip()))


def objfunx(x, vector, CR, k_list):  # 新增k_list参数
    x1, x2, x3, x4 = x
    matrix = np.array([
        [x1, 1 - x1, 0, 0, 0],
        [0, x2, 1 - x2, 0, 0],
        [0, 0, x3, 1 - x3, 0],
        [0, 0, 0, x4, 1 - x4],
        [0, 0, 0, 0, 1]
    ], dtype=float)

    target = np.array([1, 2, 3, 4, 5])
    vector = np.array(vector, dtype=float)

    total = 0.0
    for k, cr in zip(k_list, CR):  # 直接遍历k_list和CR的对应值
        matrix_power = np.linalg.matrix_power(matrix, k)
        predicted = vector @ matrix_power @ target
        total += (predicted - cr) ** 2  # 直接使用cr

    return total

def markov_regressive(CR, vector, k_list):
    # 参数校验
    assert len(CR) == len(k_list), "CR和k_list长度必须一致"
    assert all(k > 0 for k in k_list), "k_list中的值必须为整数且大于0"

    # 定义边界和约束
    bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])
    linear_constraint = LinearConstraint(
        np.eye(4),
        lb=[0.5, 0.5, 0.5, 0.5],
        ub=np.inf * np.ones(4)
    )

    # 初始猜测
    x0 = [0.8, 0.8, 0.8, 0.8]

    # 执行优化（通过args传递k_list）
    result = minimize(
        objfunx,
        x0,
        args=(vector, CR, k_list),  # 新增k_list参数
        method="trust-constr",
        bounds=bounds,
        constraints=[linear_constraint],
        options={"verbose": 1, "gtol": 1e-6, "maxiter": 1000}
    )

    # 构建返回矩阵
    x1, x2, x3, x4 = result.x
    matrix = np.array([
        [x1, 1-x1, 0,    0,    0],
        [0,  x2,  1-x2, 0,    0],
        [0,  0,   x3,  1-x3, 0],
        [0,  0,   0,    x4,  1-x4],
        [0,  0,   0,    0,    1]
    ], dtype=float)

    return matrix

class ActorCriticApp(QMainWindow,):
    def __init__(self,
                 Actor,
                 Critic,
                 Environment_set):
        super().__init__()
        # 添加组件参数配置
        self.component_params = {
            1: {'lower': [8, 10, 13.33, 8.67], 'upper': [16, 20, 26.67, 17.33]},
            2: {'lower': [6, 7.8, 9.6, 6.6], 'upper': [14, 18.2, 22.4, 15.4]},
            3: {'lower': [2, 3, 3, 2], 'upper': [6, 9, 9, 6]},
            4: {'lower': [6, 7.5, 9, 7.5], 'upper': [10, 12.5, 15, 12.5]},
            5: {'lower': [8, 11.2, 12.8, 8], 'upper': [12, 16.8, 19.2, 12]},
            6: {'lower': [6.93, 13.87, 12, 7.2], 'upper': [10.4, 20.8, 18, 10.8]},
            7: {'lower': [3.47, 6.93, 6, 3.6], 'upper': [6.93, 13.87, 12, 7.2]},
            8: {'lower': [5.25, 10.5, 12.25, 7], 'upper': [6.75, 13.5, 15.75, 9]},
            9: {'lower': [4.72, 15, 21.94, 8.33], 'upper': [16.61, 21, 30.72, 11.67]},
            10: {'lower': [1.89, 6, 8.78, 3.33], 'upper': [5.67, 18, 26.33, 10]},
            11: {'lower': [2, 2.67, 3.33, 2], 'upper': [4, 5.33, 6.67, 4]},
            12: {'lower': [9.6, 10.4, 10.4, 9.6], 'upper': [14.4, 15.6, 15.6, 14.4]},
            13: {'lower': [2.5, 3, 2.5, 2], 'upper': [10, 12, 10, 8]},
            14: {'lower': [12.86, 12.86, 17.14, 17.14], 'upper': [17.14, 17.14, 22.86, 22.86]},
            15: {'lower': [1.17, 1.75, 2.33, 1.75], 'upper': [2.83, 4.25, 5.67, 4.25]},
            16: {'lower': [6.67, 10.67, 13.33, 9.33], 'upper': [13.33, 21.33, 26.67, 18.67]},
            17: {'lower': [1.5, 1, 1.5, 1], 'upper': [4.5, 3, 4.5, 3]},
            18: {'lower': [4.1, 10, 13.2, 12.7], 'upper': [6.15, 15, 19.8, 19.05]},
            19: {'lower': [5, 6.25, 7.5, 6.25], 'upper': [11, 13.75, 16.5, 13.75]},
            20: {'lower': [6.67, 10, 13.33, 10], 'upper': [13.33, 20, 26.67, 20]}
        }
        self.protection = [0, 15, 10, 4, 12, 15, 16, 8, 9, 10, 10,
                           4, 16, 3, 20, 4, 7, 4, 15, 12, 17]
        self.actor_model = Actor
        self.critic_model = Critic
        self.initUI()
        self.load_models()
        self.Environment = Environment_set

    def update_plot(self):
        """更新概率分布图"""
        try:
            values = [float(le.text()) for le in self.struct_state_inputs]

            self.ax.clear()

            # 绘制柱状图
            bars = self.ax.bar(
                [f"S{i + 1}" for i in range(5)],
                values,
                color=['#4C72B0' if v <= 1 else '#C44E52' for v in values],
                width=0.6,
                edgecolor='white'
            )

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width() / 2,
                             height + 0.02,
                             f'{height:.2f}',
                             ha='center', va='bottom',
                             fontsize=10,
                             fontname='Times New Roman',
                             color='black')

            # 设置样式
            self.ax.set_ylim(0, 1.0)
            self.ax.set_title("State Probability Distribution",
                              fontname='Times New Roman',
                              fontsize=10)
            self.ax.set_xlabel("State", fontname='Times New Roman')
            self.ax.set_ylabel("Probability", fontname='Times New Roman')
            self.ax.grid(True, linestyle=':', alpha=0.5)

            # --- 新增：设置 X 轴和 Y 轴刻度字体 ---
            # 方法 1：单独设置 X 轴和 Y 轴
            for label in self.ax.get_xticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(10)  # 可选：调整字号

            for label in self.ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(10)  # 可选：调整字号

            # 调整子图区域（参数范围 0~1，left/bottom/right/top 控制边距）
            self.figure.subplots_adjust(left=0.11, right=0.95, bottom=0.16, top=0.86)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))

    def get_current_transition_matrix(self):
        """从界面获取当前转移矩阵"""
        matrix = np.zeros((5, 5))
        for row in range(5):
            for col in range(5):
                item = self.matrix_table.item(row, col)
                if item and item.text().strip():
                    matrix[row][col] = float(item.text())
        return matrix

    def calculate_state_evolution(self):
        """计算并绘制状态演化"""
        try:
            # 获取初始状态
            initial_state = np.array([float(le.text()) for le in self.struct_state_inputs])

            # 获取转移矩阵
            transition_matrix = self.get_current_transition_matrix()

            # 计算10年演化
            years = 10
            states = [initial_state]
            for _ in range(years - 1):
                states.append(states[-1] @ transition_matrix)

            # 转换为百分比
            states = np.array(states) * 100

            # 清除旧图
            self.evolution_ax.clear()

            # 颜色定义
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
            labels = [f'State {i + 1}' for i in range(5)]

            # 绘制堆叠柱状图
            bottom = np.zeros(years)
            for state_idx in range(5):
                values = [state[state_idx] for state in states]
                self.evolution_ax.bar(
                    range(years), values,
                    bottom=bottom,
                    color=colors[state_idx],
                    edgecolor='white',
                    label=labels[state_idx],
                    width=0.8
                )
                bottom += values

            # 设置图表样式
            self.evolution_ax.set_xticks(range(years))
            self.evolution_ax.set_xticklabels([f'{i + 1}' for i in range(years)], fontname='Times New Roman')
            self.evolution_ax.set_ylim(0, 100)
            for label in self.evolution_ax.get_yticklabels():
                label.set_fontname("Times New Roman")
                label.set_fontsize(12)  # 可选：同步设置字号
            self.evolution_ax.legend(
                loc='upper right',
                bbox_to_anchor=(1.15, 1),
                prop={'family': 'Times New Roman'}
            )

            self.evolution_canvas.draw()

        except Exception as e:
            print(f"Evolution error: {str(e)}")

    def update_transition_matrix(self):
        """更新转移矩阵显示"""
        try:
            # 获取输入值
            a_values = [float(le.text()) if le.text() else 0.0 for le in self.a_inputs]

            # 生成矩阵
            matrix = np.zeros((5, 5))
            for i in range(4):
                denominator = 1 + a_values[i]
                if denominator == 0:
                    continue
                matrix[i][i] = a_values[i] / denominator
                matrix[i][i + 1] = 1 / denominator
            matrix[4][4] = 1.0

            # 更新表格
            for row in range(5):
                for col in range(5):
                    value = matrix[row][col]
                    item = QTableWidgetItem()
                    item.setText(f"{value:.4f}" if value != 0 else "0")
                    item.setTextAlignment(Qt.AlignCenter)

                    # 设置颜色标记
                    if value > 0.5:
                        item.setBackground(QColor(200, 255, 200))
                    elif value > 0:
                        item.setBackground(QColor(255, 255, 200))

                    self.matrix_table.setItem(row, col, item)

            self.calculate_state_evolution()
        except Exception as e:
            print(f"Update error: {str(e)}")

    def create_component_id_widget(self):
        """创建构件ID选择组件"""
        # 构件类型列表
        components = [
            ("1: concrete_bridge", 1),
            ("2: steel_bridge", 2),
            ("3: deck", 3),
            ("4: superstructure", 4),
            ("5: substructure", 5),
            ("6: RC_girder", 6),
            ("7: slab", 7),
            ("8: diaphragm", 8),
            ("9: arch", 9),
            ("10: transverse", 10),
            ("11: hanger", 11),
            ("12: spandrel", 12),
            ("13: support", 13),
            ("14: tower", 14),
            ("15: cable", 15),
            ("16: steel_girder", 16),
            ("17: pavement", 17),
            ("18: column", 18),
            ("19: abutment", 19),
            ("20: foundation", 20)
        ]

        # 创建组件容器
        component_group = QGroupBox("Component ID Selection")
        layout = QVBoxLayout()

        # 创建下拉菜单
        self.component_combo = QComboBox()
        self.component_combo.setStyleSheet("""
            QComboBox {
                font: 10pt 'Times New Roman';
                min-width: 200px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                font: 10pt 'Times New Roman';
                min-width: 250px;
            }
        """)

        # 添加选项
        for text, data in components:
            self.component_combo.addItem(text, userData=data)

        # 添加标签
        label = QLabel("Select Component Type:")
        label.setStyleSheet("font: 10pt 'Times New Roman'; margin-bottom: 8px;")

        # 当前选择显示
        self.current_selection = QLabel("Selected: 1 - concrete_bridge")
        self.current_selection.setStyleSheet("""
            QLabel {
                font: 9pt 'Times New Roman';
                color: #666;
                margin-top: 8px;
            }
        """)

        # 信号连接
        self.component_combo.currentIndexChanged.connect(self.update_parameters_by_component)
        self.component_combo.currentIndexChanged.connect(self.update_component_selection)  # 保持原有功能

        # 组装布局
        layout.addWidget(label)
        layout.addWidget(self.component_combo, alignment=Qt.AlignCenter)
        layout.addWidget(self.current_selection, alignment=Qt.AlignCenter)
        component_group.setLayout(layout)

        return component_group

    def update_component_selection(self):
        """更新当前选择的构件显示"""
        current_text = self.component_combo.currentText()
        component_id = self.component_combo.currentData()
        self.current_selection.setText(
            f"Selected: {current_text.split(':')[0].strip()} - {current_text.split(':')[1].strip()}")

    def validate_corrosion(self):
        """防腐时间验证（修复版）"""
        try:
            component_id = self.component_combo.currentData()
            if not (1 <= component_id <= 20):
                return

            value = self.corrosion_spin.value()
            max_allowed = min(self.protection[component_id], 20)

            # 清除旧警告
            self.corrosion_warning.setText("")

            if value > max_allowed:
                self.corrosion_warning.setText(f"超过构件限制! (最大允许: {max_allowed}年)")
            elif value > 20:
                self.corrosion_warning.setText("超过系统限制! (最大允许: 20年)")

        except Exception as e:
            print(f"验证错误: {str(e)}")

    def update_corrosion_validation(self):
        """更新防腐时间验证规则（修复版）"""
        try:
            component_id = self.component_combo.currentData()
            if not (1 <= component_id <= 20):
                return

            # 获取保护时间并限制最大值
            protection = self.protection[component_id]
            max_allowed = min(protection, 20)

            # 更新输入范围
            self.corrosion_spin.setMaximum(max_allowed)

            # 更新提示标签
            self.corrosion_limit_label.setText(
                f"Max allowed: {max_allowed}year\n(Component recommend: {protection}year)" if protection < 20
                else f"Max allowed: 20年 (system limited)"
            )

            # 强制验证当前值
            self.validate_corrosion()

        except Exception as e:
            print(f"更新验证规则错误: {str(e)}")

    def update_parameters_by_component(self):
        # 获取当前选择的构件ID
        component_id = self.component_combo.currentData()

        # 获取对应的参数范围
        params = self.component_params.get(component_id)

        # 遍历四个参数输入框
        for i in range(4):
            lower = params['lower'][i]
            upper = params['upper'][i]

            # 设置默认值为范围中值
            default = round((lower + upper) / 2, 2)
            self.a_inputs[i].setText(str(default))

            # 创建新的验证器
            validator = QDoubleValidator(lower, upper, 2)
            validator.setNotation(QDoubleValidator.StandardNotation)
            self.a_inputs[i].setValidator(validator)

            # 设置悬浮提示
            self.a_inputs[i].setToolTip(f"Allowed range: {lower} ~ {upper}")

        # 更新防腐时间限制
        self.update_corrosion_validation()

    def Monte_Carlo_Simulation(self):

        inputs = self.process_inputs()
        Action_number = np.zeros((5, 1), dtype=float)

        for i in range(100):
            _, actions_pred, _, _ = self.Environment.future_estimation(
                inputs, self.actor_model)
            actions_int = actions_pred.flatten().astype(int)
            counts = np.bincount(actions_int, minlength=5)  # 确保长度为5
            Action_number += counts.reshape(-1, 1)  # 累加到总计数

        Action_number /= 100
        return Action_number

    # === 预测执行方法 ===
    def run_prediction(self):
        """执行预测并更新显示"""
        try:
            # 获取输入数据（需替换为实际数据获取逻辑）
            inputs = self.process_inputs()

            # 模拟预测结果
            logit = self.actor_model(inputs)
            actor_probs = tf.nn.softmax(logit).numpy().reshape(-1)
            critic_value = self.critic_model(inputs).numpy().squeeze()

            # 更新Actor显示
            self.update_actor_plot(actor_probs)

            # 更新Critic显示
            self.critic_value.setText(f"{critic_value:.4f}")

            # 执行预测（假设 env 和 actor_net 已初始化）
            states_pred, actions_pred, rewards_pred, hidden_pred = self.Environment.future_estimation(
                inputs, self.actor_model
            )

            # === 更新状态变化图 ===
            # 原先的绘图轴
            self.state_evolution_ax.clear()

            # 创建双Y轴：左侧(ax1)用于状态概率的堆叠柱状图，右侧(ax2)用于期望状态值折线
            ax1 = self.state_evolution_ax
            ax2 = ax1.twinx()  # 共享x轴，双Y轴

            # 设置全局字体
            plt.rcParams['font.family'] = 'Times New Roman'

            # 提取状态概率(假设前5列为分布)
            state_probs = states_pred[:, :5]  # shape: (T, 5)
            # 计算期望状态
            expected_state = state_probs @ np.array([1, 2, 3, 4, 5])

            # x 轴坐标
            x = np.arange(state_probs.shape[0])

            # 定义颜色
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

            # 在 ax1 上绘制累积柱状图(状态分布)，并为每个状态添加 label
            bottom_array = np.zeros_like(state_probs[:, 0])
            for i in range(5):
                ax1.bar(
                    x,
                    state_probs[:, i],
                    bottom=bottom_array,
                    color=colors[i],
                    alpha=0.6,
                    label=f"State {i + 1}"  # 每个状态都添加图例
                )
                bottom_array += state_probs[:, i]

            # 在 ax2 上绘制期望状态值的折线图
            ax2.plot(
                x,
                expected_state,
                color='black',
                linewidth=2,
                label='Expected State'
            )

            # 在期望值线上标注动作位置
            action_markers = {
                1: {'marker': '^', 'color': 'red', 'size': 24, 'label': 'NDT'},
                2: {'marker': 's', 'color': 'green', 'size': 24, 'label': 'Vis+Prev'},
                3: {'marker': 'D', 'color': 'blue', 'size': 24, 'label': 'NDT+Prev'},
                4: {'marker': '*', 'color': 'purple', 'size': 32, 'label': 'Replace'}
            }
            added_labels = set()
            for step in range(len(actions_pred)):
                action = actions_pred[step]
                if action in action_markers:
                    mkinfo = action_markers[action]
                    ax2.scatter(
                        step,
                        expected_state[step],
                        marker=mkinfo['marker'],
                        color=mkinfo['color'],
                        s=mkinfo['size'],
                        edgecolors='black',
                        linewidths=0.8,
                        zorder=20
                    )
                    # 只为每种动作类型添加一次图例
                    if action not in added_labels:
                        ax2.scatter(
                            [],
                            [],
                            marker=mkinfo['marker'],
                            color=mkinfo['color'],
                            s=mkinfo['size'],
                            label=mkinfo['label']
                        )
                        added_labels.add(action)

            # 设置左侧y轴范围 (0 ~ 1)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("State Probability (0-1)")

            # 设置右侧y轴范围 (1 ~ 5)
            ax2.set_ylim(1, 5)
            ax2.set_ylabel("Expected State (1-5)")

            # 设置x轴标签(可选)
            ax1.set_xlabel("Time Step")

            # 组合双Y轴的图例，并在 ax1 上统一显示
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # === 字体设置统一配置 ===
            # 设置坐标轴标签
            self.state_evolution_ax.set_xlabel("Time Steps",
                                               fontdict={'family': 'Times New Roman', 'size': 12})
            self.state_evolution_ax.set_ylabel("State Probability",
                                               fontdict={'family': 'Times New Roman', 'size': 12})

            # 设置刻度字体
            for label in self.state_evolution_ax.get_xticklabels() + self.state_evolution_ax.get_yticklabels():
                label.set_fontproperties('Times New Roman')
                label.set_fontsize(10)

            # 设置图例字体
            legend = self.state_evolution_ax.legend(
                loc='upper right',
                prop={'family': 'Times New Roman', 'size': 9},
                frameon=True,
                framealpha=0.8
            )

            # 强制应用字体设置（解决某些系统字体缓存问题）
            plt.setp(legend.get_texts(), fontfamily='Times New Roman')

            # 设置刻度间隔
            self.state_evolution_ax.set_xticks(range(0, 20, 2))

            # === 图表重绘 ===
            self.state_evolution_canvas.draw()

            # 获取动作执行次数数据
            Action_number = self.Monte_Carlo_Simulation()  # 蒙特卡洛模拟方法

            # 更新柱状图
            self.action_ax.clear()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            bars = self.action_ax.bar(range(5), Action_number.flatten(),
                                      color=colors, edgecolor='black')

            # === 添加数值标签 ===
            for bar in bars:
                height = bar.get_height()
                self.action_ax.text(
                    x=bar.get_x() + bar.get_width() / 2,  # 居中位置
                    y=height + 0.02,  # 略高于柱顶
                    s=f'{height:.2f}',  # 显示整数值
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontname='Times New Roman',
                    fontsize=10,
                    color='black'
                )

            self.action_ax.set_ylim(0, 19.0)
            self.action_ax.set_title("Action Execution Counts", fontname='Times New Roman')
            self.action_ax.set_xticks(range(5))
            self.action_ax.set_xticklabels(["VisIns", "NDT", "Vis+Prev", "NDT+Prev", "Replace"])
            self.action_count_canvas.draw()

            self.statusBar().showMessage("Prediction Completed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"预测失败: {str(e)}")

    def update_actor_plot(self, probabilities):
        """更新Actor柱状图"""
        self.actor_ax.clear()

        # 定义动作标签和颜色
        actions = ["VisIns", "NDT", "Vis+Prev", "NDT+Prev", "Replace"]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']

        # 绘制柱状图
        bars = self.actor_ax.bar(actions, probabilities, color=colors)

        # 设置图表样式
        self.actor_ax.set_ylim(0, 1.1)
        self.actor_ax.set_title("Action Probability Distribution", fontsize=12, fontname='Times New Roman')
        self.actor_ax.tick_params(axis='x', rotation=0, labelsize=8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            self.actor_ax.text(bar.get_x() + bar.get_width() / 2.,
                               height + 0.02,
                               f'{height:.2f}',
                               ha='center',
                               va='bottom',
                               fontsize=10,
                               fontname='Times New Roman')

        # 设置 X 轴和 Y 轴刻度字体
        for label in self.actor_ax.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(12)  # 与 tick_params 中的 labelsize 一致

        for label in self.actor_ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(12)

        # 调整子图边距（参数范围 0~1，按需调整）
        self.actor_figure.subplots_adjust(
            left=0.1,  # 左边距
            right=0.95,  # 右边距
            bottom=0.2,  # 下边距（为X轴标签留空间）
            top=0.9  # 上边距
        )

        self.actor_canvas.draw()

    def import_data_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Data File",
            filter="Excel Files (*.xlsx);;CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            # 读取时明确指定无表头
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl', header=None)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
            else:
                QMessageBox.critical(self, "Error", "Unsupported file format")
                return

            # 增强数据验证
            if df.shape[1] != 6:
                QMessageBox.critical(
                    self, "Error",
                    f"Need 6 column data，current check {df.shape[1]} column\n"
                    f"First two data example:\n{df.head(2).to_string(header=False)}"
                )
                return

            # 验证首列是否为整数（年份）
            if not pd.api.types.is_integer_dtype(df.iloc[:, 0]):
                QMessageBox.critical(
                    self, "Error",
                    "First column should be integer\n"
                    f"Check the data type as: {df.iloc[:, 0].dtype}"
                )
                return

            self.data_table.setRowCount(0)

            # 使用itertuples提升性能
            for row in df.itertuples(index=False):
                row_idx = self.data_table.rowCount()
                self.data_table.insertRow(row_idx)

                # 处理时间列（第0列）
                time_item = QTableWidgetItem(str(row[0]))
                time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(row_idx, 0, time_item)

                # 处理数据列（1-5列）
                for col in range(1, 6):
                    try:
                        value = str(row[col])
                        # 可添加数值格式验证
                        if not value.isdigit():
                            raise ValueError(f"No.{col} contain str data")
                    except Exception as e:
                        QMessageBox.warning(
                            self, "Data Warning",
                            f"No.{row_idx + 1} row data error，replaced by\n{str(e)}"
                        )
                        value = ""

                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.data_table.setItem(row_idx, col, item)

            QMessageBox.information(
                self, "Success",
                f"Successful load {len(df)} row data\n"
                f"time range: {df.iloc[:, 0].min()} - {df.iloc[:, 0].max()}"
            )


        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import data:\n{str(e)}")
    def create_editable_table(self):
        # 表格和按钮的容器
        table_container = QWidget()
        layout = QHBoxLayout(table_container)

        # 创建表格（保持原有设置）
        self.data_table = EditableTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(['Inspection time', 'State1', 'State2',
                                                   'State3', 'State4', 'State5'])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setMinimumHeight(150)

        # 按钮容器（右侧垂直排列）
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(10, 0, 0, 0)

        # 修改后的 Add Data 按钮
        btn_add = QPushButton("Import Data")
        btn_add.setFixedWidth(120)
        btn_add.clicked.connect(self.import_data_from_file)  # 绑定新方法

        btn_clear = QPushButton("Clear Table")
        btn_clear.setFixedWidth(120)
        btn_clear.clicked.connect(lambda: self.data_table.setRowCount(0))

        # 添加弹簧控制垂直位置
        btn_layout.addStretch(1)  # 顶部弹簧
        btn_layout.addWidget(btn_add, alignment=Qt.AlignmentFlag.AlignHCenter)
        btn_layout.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignHCenter)
        btn_layout.addStretch(1)  # 底部弹簧

        layout.addWidget(self.data_table, stretch=5)
        layout.addWidget(btn_container, stretch=1)

        return table_container

    def create_plot_widget(self):
        # 绘图容器
        plot_group = QGroupBox("Expected state changing")
        main_layout = QHBoxLayout(plot_group)  # 主布局仍为水平
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)  # 设置组件间距

        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']  # 设置主要字体
        rcParams['font.size'] = 12  # 基础字号
        rcParams['axes.titlesize'] = 14  # 坐标轴标题字号
        rcParams['axes.labelsize'] = 13  # 坐标轴标签字号
        rcParams['xtick.labelsize'] = 12  # X轴刻度字号
        rcParams['ytick.labelsize'] = 12  # Y轴刻度字号
        rcParams['legend.fontsize'] = 12  # 图例字号

        # ------------------------- 图表部分 -------------------------
        # 用容器包裹图表以便单独控制占比
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)  # 移除内部边距

        # Matplotlib组件（保持原有配置）
        self.figure_CR = Figure()
        self.canvas_CR = FigureCanvas(self.figure_CR)
        self.ax_CR = self.figure_CR.add_subplot(111)
        self.canvas_CR.setMinimumHeight(200)

        # 初始化空图表样式（保持原有配置）
        self.ax_CR.grid(True, linestyle='--', alpha=0.7)
        self.ax_CR.set_xlabel("Inspection Index")
        self.ax_CR.set_ylabel("CR Value")

        plot_layout.addWidget(self.canvas_CR)

        # 将图表容器添加到主布局，并设置占比为4（可调整）
        main_layout.addWidget(plot_container, stretch=5)  # 图表占4/5空间

        # ------------------------- 按钮部分 -------------------------
        # 按钮容器用于垂直居中
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)

        btn_update = QPushButton("Update Plot")
        btn_update.setFixedWidth(120)  # 固定宽度
        btn_update.clicked.connect(self.update_state_plot)

        # 添加弹簧控制垂直位置
        button_layout.addStretch(1)  # 顶部弹簧
        button_layout.addWidget(btn_update, alignment=Qt.AlignmentFlag.AlignHCenter)
        button_layout.addStretch(1)  # 底部弹簧

        # 将按钮容器添加到主布局，并设置占比为1（可调整）
        main_layout.addWidget(button_container, stretch=1)  # 按钮占1/5空间

        return plot_group

    def create_markov_table(self):
        # Markov矩阵表格容器
        matrix_group = QGroupBox("Regressive state transition matrix")
        main_layout = QHBoxLayout(matrix_group)  # 主布局改为水平
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)  # 设置组件间距

        # ------------------------- 表格部分 (占5/6空间) -------------------------
        # 表格容器控制占比
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        # 创建表格控件
        self.markov_table_Regressive = QTableWidget()
        self.markov_table_Regressive.setRowCount(5)
        self.markov_table_Regressive.setColumnCount(5)
        self.markov_table_Regressive.setHorizontalHeaderLabels(['S1', 'S2', 'S3', 'S4', 'S5'])
        self.markov_table_Regressive.setVerticalHeaderLabels(['S1', 'S2', 'S3', 'S4', 'S5'])
        self.markov_table_Regressive.setEditTriggers(QTableWidget.NoEditTriggers)

        # 设置表头自适应
        self.markov_table_Regressive.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.markov_table_Regressive.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 设置表头字体居中
        self.markov_table_Regressive.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.markov_table_Regressive.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        for row in range(5):
            for col in range(5):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)  # 设置单元格内容居中
                self.markov_table_Regressive.setItem(row, col, item)

        self.markov_table_Regressive.setMaximumHeight(200)
        table_layout.addWidget(self.markov_table_Regressive)

        # 添加表格容器到主布局（占比5）
        main_layout.addWidget(table_container, stretch=5)

        # ------------------------- 按钮部分 (占1/6空间) -------------------------
        # 按钮容器控制垂直居中
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        # 新增按钮
        btn_update = QPushButton("Update Matrix")
        btn_update.clicked.connect(self.update_markov_matrix)
        btn_update.setFixedHeight(35)  # 固定高度
        btn_update.setFixedWidth(120)  # 新增固定宽度

        # 添加弹簧使按钮垂直居中
        button_layout.addStretch(1)
        button_layout.addWidget(btn_update, alignment=Qt.AlignmentFlag.AlignHCenter)
        button_layout.addStretch(1)

        # 添加按钮容器到主布局（占比1）
        main_layout.addWidget(button_container, stretch=1)

        return matrix_group

    def update_state_plot(self):
        try:
            cr_values = []
            x_labels = []  # 新增：存储第一列的时间数据
            initial_vector = None  # 新增：初始状态向量
            for row in range(self.data_table.rowCount()):
                states = []
                valid_row = True
                time_value = None  # 新增：存储当前行的时间值

                # 第一步：读取时间列（第0列）
                time_item = self.data_table.item(row, 0)
                if time_item and time_item.text().isdigit():
                    time_value = int(time_item.text())
                else:
                    valid_row = False

                # 第二步：验证状态列（1-5列）
                for col in range(1, 6):
                    item = self.data_table.item(row, col)
                    if item and item.text().isdigit():
                        states.append(int(item.text()))
                    else:
                        valid_row = False
                        break

                # 只有时间和状态都有效才记录数据
                if valid_row and time_value is not None:
                    numerator = sum((i + 1) * states[i] for i in range(5))
                    denominator = sum(states)
                    cr = numerator / denominator if denominator != 0 else 0
                    cr_values.append(cr)
                    x_labels.append(time_value)  # 记录有效时间值

                    # 提取初始向量（仅第一行有效数据）
                    if initial_vector is None and denominator != 0:
                        # 归一化处理为概率分布
                        total = sum(states)
                        initial_vector = [s / total for s in states]

            # 存储为类属性
            self.cr_values = cr_values  # 存储CR序列
            self.x_labels = x_labels  # 存储时间标签
            self.initial_vector = initial_vector  # 存储初始向量

            # 绘图逻辑
            self.ax_CR.clear()
            if cr_values:
                # 使用实际时间数据作为x轴
                self.ax_CR.plot(x_labels, cr_values, 'b-^',  # x轴改为x_labels
                                linewidth=2,
                                markersize=8,
                                label='CR Trend')

                # 设置带时间标签的x轴
                self.ax_CR.set_xlabel("Inspection Year", fontsize=12)  # 修改标签名称
                self.ax_CR.set_ylabel("CR Value", fontsize=12)

                # 自动生成合适的刻度间隔
                self.ax_CR.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # 强制显示整数刻度
                self.ax_CR.grid(True, linestyle='--', alpha=0.7)
                self.ax_CR.legend()

                # 优化坐标范围
                self.ax_CR.set_xlim(min(x_labels) - 0.5, max(x_labels) + 0.5)  # 动态范围
                self.ax_CR.autoscale(axis='y')

                # 添加数据标签（可选）
                for x, y in zip(x_labels, cr_values):
                    self.ax_CR.text(x, y, f'{y:.2f}',
                                    ha='center', va='bottom',
                                    fontsize=10)
            else:
                self.ax_CR.text(0.5, 0.5, "No valid data",
                                transform=self.ax_CR.transAxes,
                                ha='center', va='center',
                                fontsize=14)

            # 优化布局
            self.figure.tight_layout()
            self.canvas_CR.draw_idle()


        except Exception as e:
            QMessageBox.critical(self, "Critical Error",f"绘图失败:\n{str(e)}")

    def update_markov_matrix(self):
        try:
            # 检查数据是否存在
            if not hasattr(self, 'cr_values') or not self.cr_values:
                raise ValueError("请先更新CR趋势图以生成数据")
            if not hasattr(self, 'x_labels') or not self.x_labels:
                raise ValueError("未找到时间数据")
            if not hasattr(self, 'initial_vector') or not self.initial_vector:
                raise ValueError("未找到初始状态向量")

            # 生成k_list（时间步长）
            min_time = min(self.x_labels)  # 起始时间
            k_list = [t - min_time + 1 for t in self.x_labels]  # 转换为步长（如2010→1, 2011→2）

            # 调用Markov回归计算
            matrix = markov_regressive(
                CR=self.cr_values,
                vector=self.initial_vector,
                k_list=k_list
            )

            # 更新Markov矩阵表格
            for i in range(5):
                for j in range(5):
                    value = f"{matrix[i][j]:.3f}"
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.markov_table_Regressive.setItem(i, j, item)

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"更新失败:\n{str(e)}")

    def initUI(self):
        self.setWindowTitle('Bridge management System V1.0')
        self.setGeometry(100, 100, 1100, 800)

        # 总体样式设置：简化 + 去除过多 margin-top，避免标题被遮挡
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            /* 页签面板 */
            QTabWidget::pane {
                background: #FFFFFF;
                border: 1px solid #C6C6C6;
                border-radius: 5px;
                margin: 0px;
            }
            /* 顶部页签 */
            QTabBar::tab {
                background: #E2E2E2;
                font: 10pt 'Times New Roman';
                padding: 6px 16px;
                border: 1px solid #C6C6C6;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: -1px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
            }
            /* 分组框样式：去除多余 margin-top，增加内边距padding避免标题与内容重叠 */
            QGroupBox {
                border: 1px solid #C6C6C6;
                border-radius: 6px;
                padding: 10px; /* 内部留一些空间以免标题和内容贴边 */
                font: bold 10pt 'Times New Roman';
                background-color: #FAFAFA;
                margin-top: 6px; /* 整体和外面保持一些间距 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background-color: transparent;
                left: 14px;
                top: 4px;
                padding: 0 3px;
            }

            QLabel {
                font: 10pt 'Times New Roman';
                color: #333333;
            }
            QLineEdit {
                font: 10pt 'Times New Roman';
                border: 1px solid #C6C6C6;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton {
                font: 10pt 'Times New Roman';
                background-color: #0078D4;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #005EA8;
            }
            QSpinBox {
                font: 10pt 'Times New Roman';
                border: 1px solid #C6C6C6;
                border-radius: 3px;
                padding: 4px;
            }
            QTableWidget {
                background-color: #FFFFFF;
                gridline-color: #E0E0E0;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                font: 10pt 'Times New Roman';
                padding: 4px;
                border: 1px solid #D0D0D0;
            }
            QStatusBar {
                font: 9pt 'Times New Roman';
            }
        """)

        # 主控件 + 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        # 适当缩小边距和间距，使界面更紧凑
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        # 页签组件
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # ------------------ 第三个分页 tab3 ------------------
        self.tab3 = QWidget()

        # 主布局
        self.tab3_layout = QVBoxLayout(self.tab3)
        self.tab3_layout.setContentsMargins(10, 10, 10, 10)
        self.tab3_layout.setSpacing(15)

        # 1. 创建可编辑表格（含按钮）
        table_container = self.create_editable_table()
        # 2. 创建绘图区域
        plot_group = self.create_plot_widget()
        # 3. 创建Markov矩阵显示
        matrix_group = self.create_markov_table()

        # 为了更好地控制“中间的图”不至于过大，可以使用 stretch 参数
        # 让三部分在垂直方向上按指定权重分配空间
        self.tab3_layout.addWidget(table_container, stretch=1)
        self.tab3_layout.addWidget(plot_group, stretch=1)
        self.tab3_layout.addWidget(matrix_group, stretch=2)

        self.tab_widget.addTab(self.tab3, "Markov chain")


        # ------------------ 第一个分页 tab1 ------------------
        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        # 同样缩小分页布局的边距和间距
        tab1_layout.setContentsMargins(10, 10, 10, 10)
        tab1_layout.setSpacing(10)

        # “腐蚀+构件”容器
        corrosion_container = QWidget()
        corrosion_container_layout = QHBoxLayout(corrosion_container)
        corrosion_container_layout.setContentsMargins(0, 0, 0, 0)
        corrosion_container_layout.setSpacing(20)

        # 腐蚀时间模块
        corrosion_group = QGroupBox("Corrosion Time (0-20 years)")
        corrosion_layout = QVBoxLayout()
        self.corrosion_spin = QSpinBox()
        self.corrosion_spin.setRange(0, 20)
        self.corrosion_spin.setValue(0)
        self.corrosion_spin.setStyleSheet("""
            QSpinBox {
                font: 12pt 'Times New Roman';
                padding: 5px;
                min-width: 120px;
                margin: 10px;
            }
        """)
        self.corrosion_limit_label = QLabel("Max allowed: 20 years")
        self.corrosion_limit_label.setStyleSheet("""
            font: 9pt 'Times New Roman';
            color: #666;
        """)
        self.corrosion_warning = QLabel("")
        self.corrosion_warning.setStyleSheet("""
            color: red;
            font: 10pt 'Times New Roman';
            min-height: 20px;
            margin-bottom: 10px;
        """)
        input_container = QWidget()
        input_container_layout = QVBoxLayout(input_container)
        input_container_layout.setSpacing(5)
        input_container_layout.addWidget(QLabel("Years:"), alignment=Qt.AlignCenter)
        input_container_layout.addWidget(self.corrosion_spin, alignment=Qt.AlignCenter)
        input_container_layout.addWidget(self.corrosion_limit_label, alignment=Qt.AlignCenter)
        input_container_layout.addWidget(self.corrosion_warning, alignment=Qt.AlignCenter)
        corrosion_layout.addWidget(input_container)
        corrosion_group.setLayout(corrosion_layout)
        # 让布局更灵活，不固定宽度
        # corrosion_group.setFixedWidth(300)

        # 构件ID模块
        component_group = self.create_component_id_widget()
        # 同样不固定太宽，让布局自动适配
        # component_group.setFixedWidth(320)

        # 信号连接（不可去除）
        self.component_combo.currentIndexChanged.connect(self.update_corrosion_validation)
        self.corrosion_spin.valueChanged.connect(self.validate_corrosion)

        # 布局组合
        corrosion_container_layout.addWidget(component_group)
        corrosion_container_layout.addWidget(corrosion_group)
        tab1_layout.addWidget(corrosion_container, alignment=Qt.AlignTop)

        # “结构状态”水平布局
        align_layout = QHBoxLayout()
        align_layout.setSpacing(20)

        # 左侧 - State Inputs
        input_group = QGroupBox("State Inputs (1x5)")
        input_layout = QGridLayout()
        input_layout.setSpacing(8)
        labels = ["State 1:", "State 2:", "State 3:", "State 4:", "State 5:"]
        self.struct_state_inputs = []
        for row, text in enumerate(labels):
            label = QLabel(text)
            label.setStyleSheet("font: 10pt 'Times New Roman';")
            input_layout.addWidget(label, row, 0)

            le = QLineEdit("0.0")
            le.setFixedSize(100, 25)
            le.setStyleSheet("font: 10pt 'Times New Roman';")
            self.struct_state_inputs.append(le)
            input_layout.addWidget(le, row, 1)

            warn_label = QLabel("")
            warn_label.setFixedWidth(60)
            warn_label.setStyleSheet("color: red; font: 8pt 'Times New Roman';")
            input_layout.addWidget(warn_label, row, 2)

            def make_validator(idx, label_obj):
                def validate():
                    try:
                        val = float(self.struct_state_inputs[idx].text())
                        label_obj.setText(">1!" if val > 1 else "")
                    except:
                        label_obj.setText("Invalid!")

                return validate

            le.textChanged.connect(make_validator(row, warn_label))

        input_group.setLayout(input_layout)
        # input_group.setFixedWidth(300)  # 让布局自身决定宽度

        # 右侧 - Probability Plot
        plot_group = QGroupBox("Probability Distribution")
        plot_layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(0, 1.0)
        for t in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            t.set_fontname('Times New Roman')

        btn_plot = QPushButton("Update Plot")
        btn_plot.setFixedHeight(30)
        btn_plot.clicked.connect(self.update_plot)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(btn_plot)
        plot_group.setLayout(plot_layout)
        # plot_group.setFixedWidth(400)

        align_layout.addWidget(input_group, 30)
        align_layout.addWidget(plot_group, 70)
        tab1_layout.addLayout(align_layout)

        # 状态转移矩阵模块
        matrix_container = QWidget()
        matrix_layout = QHBoxLayout(matrix_container)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setSpacing(20)

        # 左侧参数输入区
        param_group = QGroupBox("Transition Parameters")
        param_layout = QFormLayout()
        self.a_inputs = []
        for i in range(1, 5):
            le = QLineEdit("1.0")
            le.setValidator(QDoubleValidator(0.0, 1000.0, 4))
            le.setStyleSheet("""
                font: 10pt 'Times New Roman';
                min-width: 80px;
                padding: 5px;
            """)
            le.textChanged.connect(self.update_transition_matrix)
            self.a_inputs.append(le)
            param_layout.addRow(f"α{i}:", le)
        param_group.setLayout(param_layout)
        # param_group.setFixedWidth(220)
        matrix_layout.addWidget(param_group, 20)

        # 中间矩阵显示区
        matrix_group = QGroupBox("Transition Matrix")
        self.matrix_table = QTableWidget(5, 5)
        self.matrix_table.verticalHeader().setVisible(False)
        self.matrix_table.horizontalHeader().setVisible(False)
        for i in range(5):
            self.matrix_table.setColumnWidth(i, 70)
            self.matrix_table.setRowHeight(i, 30)

        matrix_inner_layout = QVBoxLayout()
        matrix_inner_layout.addWidget(self.matrix_table)
        matrix_group.setLayout(matrix_inner_layout)
        matrix_layout.addWidget(matrix_group, 30)

        # 右侧演化图表区
        evolution_group = QGroupBox("State Evolution (10 Years)")
        self.evolution_figure = Figure(figsize=(8, 4.5))
        self.evolution_canvas = FigureCanvas(self.evolution_figure)
        self.evolution_ax = self.evolution_figure.add_subplot(111)
        self.evolution_ax.set_ylim(0, 100)
        self.evolution_ax.set_xlabel("Year", fontname='Times New Roman', fontsize=10)
        self.evolution_ax.set_ylabel("Probability (%)", fontname='Times New Roman', fontsize=10)
        self.evolution_ax.grid(True, alpha=0.3)
        self.evolution_ax.set_xticks(range(10))
        self.evolution_ax.set_xticklabels([str(i + 1) for i in range(10)], fontname='Times New Roman')

        evolution_layout = QVBoxLayout()
        evolution_layout.addWidget(self.evolution_canvas)
        evolution_group.setLayout(evolution_layout)
        matrix_layout.addWidget(evolution_group, 50)

        tab1_layout.addWidget(matrix_container)
        self.update_transition_matrix()  # 初始化矩阵

        self.tab_widget.addTab(tab1_widget, "Component info")

        # ------------------ 第二个分页 tab2 ------------------
        tab2_widget = QWidget()
        tab2_layout = QVBoxLayout(tab2_widget)
        tab2_layout.setContentsMargins(10, 10, 10, 10)
        tab2_layout.setSpacing(10)

        # 时间步骤设置模块
        time_step_group = QGroupBox("Time Step Configuration (0-99)")
        time_step_layout = QHBoxLayout()
        self.time_step_spin = QSpinBox()
        self.time_step_spin.setRange(0, 99)
        self.time_step_spin.setValue(0)
        self.time_step_spin.setStyleSheet("""
            QSpinBox {
                font: 12pt 'Times New Roman';
                padding: 5px;
                min-width: 100px;
                margin-right: 15px;
            }
        """)
        time_step_layout.addWidget(QLabel("Current Time Step:"))
        time_step_layout.addWidget(self.time_step_spin)
        time_step_layout.addStretch()
        time_step_group.setLayout(time_step_layout)

        # 预测结果模块
        upper_output_group = QGroupBox("Prediction Results")
        upper_output_layout = QHBoxLayout(upper_output_group)
        self.actor_figure = Figure(figsize=(5, 3))
        self.actor_canvas = FigureCanvas(self.actor_figure)
        self.actor_ax = self.actor_figure.add_subplot(111)
        upper_output_layout.addWidget(self.actor_canvas, 70)

        critic_container = QVBoxLayout()
        self.critic_label = QLabel("The current state value")
        self.critic_label.setAlignment(Qt.AlignCenter)
        self.critic_label.setStyleSheet("""
            QLabel {
                font: bold 14pt 'Times New Roman';
                color: #606060;
                margin-bottom: 8px;
            }
        """)
        self.critic_value = QLabel("0.0000")
        self.critic_value.setAlignment(Qt.AlignCenter)
        self.critic_value.setStyleSheet("""
            QLabel {
                font: bold 24pt 'Times New Roman';
                color: #27AE60;
                border: 2px solid #D0D0D0;
                border-radius: 5px;
                min-width: 180px;
                padding: 20px;
            }
        """)
        critic_container.addWidget(self.critic_label)
        critic_container.addWidget(self.critic_value)
        upper_output_layout.addLayout(critic_container, 30)

        down_output_group = QGroupBox("State development")
        down_output_layout = QHBoxLayout(down_output_group)

        # 左侧状态变化图
        self.state_evolution_figure = Figure(figsize=(6, 3))
        self.state_evolution_canvas = FigureCanvas(self.state_evolution_figure)
        self.state_evolution_ax = self.state_evolution_figure.add_subplot(111)
        self.state_evolution_ax.set_title("Structural State Evolution Over 20 Steps",
                                          fontname='Times New Roman')
        self.state_evolution_ax.set_xlabel("Time Step", fontname='Times New Roman')
        self.state_evolution_ax.set_ylabel("Probability", fontname='Times New Roman')
        self.state_evolution_ax.grid(True, linestyle='--', alpha=0.6)
        down_output_layout.addWidget(self.state_evolution_canvas, 60)

        # 右侧动作次数柱状图
        self.action_count_figure = Figure(figsize=(4, 3))
        self.action_count_canvas = FigureCanvas(self.action_count_figure)
        self.action_ax = self.action_count_figure.add_subplot(111)
        self.action_ax.set_title("Action Execution Counts", fontname='Times New Roman')
        self.action_ax.set_xlabel("Action Type", fontname='Times New Roman')
        self.action_ax.set_ylabel("Count", fontname='Times New Roman')
        self.action_ax.set_xticks(range(5))
        self.action_ax.set_xticklabels(['Action 1', 'Action 2', 'Action 3', 'Action 4', 'Action 5'])
        self.action_ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        down_output_layout.addWidget(self.action_count_canvas, 40)

        # 预测按钮
        btn_predict = QPushButton("Execute Prediction")
        btn_predict.setStyleSheet("""
            QPushButton {
                font: bold 12pt 'Times New Roman';
                padding: 10px;
                background-color: #0078D4;
                color: white;
                border-radius: 5px;
                min-width: 160px;
            }
            QPushButton:hover { background-color: #006CBE; }
        """)
        btn_predict.clicked.connect(self.run_prediction)

        # 组装第二分页
        tab2_layout.addWidget(time_step_group)
        tab2_layout.addWidget(upper_output_group)
        tab2_layout.addWidget(down_output_group)
        tab2_layout.addWidget(btn_predict, 0, Qt.AlignCenter)
        tab2_layout.addStretch(1)

        self.tab_widget.addTab(tab2_widget, "Decision-making Module")

        # 状态栏
        self.statusBar().showMessage("Ready")


    def load_models(self):
        """加载预训练模型"""
        try:
            # 加载Actor网络
            dump_input = tf.random.normal((1, 131))
            self.actor_model(dump_input)
            self.critic_model(dump_input)
            self.actor_model.load_weights('ActorCritic/Actor.weights.h5')
            self.critic_model.load_weights('ActorCritic/Critic.weights.h5')
            self.statusBar().showMessage("Neural network loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Neural network failure loaded: {str(e)}")

    def process_inputs(self):
        """处理所有输入数据并进行归一化"""
        # 结构状态 (1x5) - 已通过概率校验（0-1范围）
        struct_state = np.array([float(le.text()) for le in self.struct_state_inputs]).reshape(1, 5)

        # 抗腐蚀时间归一化（/20）
        corrosion_time = np.array([[float(self.corrosion_spin.value()) / 20]])  # 修改为使用spinBox的值

        # 状态矩阵归一化（/30）
        state_matrix = np.array([float(le.text()) for le in self.a_inputs]).reshape(1, 4) / 30

        # 构件ID (1x20 one-hot)
        component_id = np.eye(21)[[self.component_combo.currentData() - 1]]  # ID从1开始转为0-based索引

        # 时间步骤 (1x100 one-hot)
        time_step = np.eye(100)[[self.time_step_spin.value()]]  # 假设使用time_step_spin控件

        vector = np.concatenate(
            (struct_state, corrosion_time, state_matrix, component_id, time_step),
            axis=1
        )

        return vector


class environment():
    """this part define the bridge degradation process"""
    def __init__(self):
        self.state_number = 5
        self.normalized_time = 15

        self.deterioration_rate = np.array(
            [2, 3, 1.8, 2.3, 2.4, 2, 1.9, 2.3, 3, 2.8, 3.4, 1.9, 1.2, 1.5, 3.6, 3, 1.5, 1.9, 2.1, 2.0])
        self.protection = np.array([15, 10, 4, 12, 15, 16, 8, 9, 10, 10, 4, 16, 3, 20, 4, 7, 4, 15, 12, 17])

        # observation matrix
        self.accuracy_visual = 0.6
        self.accuracy_NDT = 0.99
        self.observation_visual = observation(self.accuracy_visual, self.state_number, Matrix_type=False)
        self.observation_NDT = observation(self.accuracy_NDT, self.state_number, Matrix_type=True)

        # defined repair or replace action state transition matrix
        self.repair_matrix = np.zeros((self.state_number, self.state_number))
        self.repair_matrix[:, 0] = 1

        # define the cost & risk value in different component,
        self.risk = np.array([0, 0, 0, -0.6, -1.5])
        self.cost = np.array([-0.01, -0.1, -0.15, -1])
        self.input_shape = 131


    def step(self, states, actions, hidden_state):
        """
        :param states: last item is time with one-hot
        :param actions: 0: visual inspection, 1: NDT, 2: preventive maintenance+visual inspection,
        3: preventive maintenance+NDT, 4: replacement
        :param hidden_state: 0-4, [3 + 20 + 4 + 44 + 63 + 6 + 6 + 6]
        :return: new_state, reward, new_hidden_state
        """
        # obtain the state information from inputting vector
        state = states.flatten()
        component_ID = tf.argmax(state[10:31]).numpy()

        component_state = state[0:5]
        protection_time = state[5] * 20  # normalization
        duration_time = state[6:10] * 30  # normalization


        time = tf.argmax(state[31:131]).numpy()

        protection_time += -1
        time += 1

        """
        Repair or replace component based on the action-----------------------------------------------------------------
        """
        cost_repair = 0
        if actions == 4:
            component_state = component_state @ self.repair_matrix
            hidden_state = 0
            protection_time = self.protection[component_ID]
            cost_repair += self.cost[3]

        """
        based protection time and duration time calculate state-transition matrix---------------------------------------
        """

        state_T_D = Markov_state_transition_matrix(self.state_number,
                                                               duration_time / self.deterioration_rate[component_ID])
        state_T = Markov_state_transition_matrix(self.state_number, duration_time)

        component_new_state, new_protection_time, new_hidden_state = state_evolution(component_state, protection_time,
                                                                                     hidden_state, state_T, state_T_D,
                                                                                     self.normalized_time)
        """
        inspection part-------------------------------------------------------------------------------------------------
        """
        cost_inspection = 0
        if actions == 0 or actions == 2:
            cost_inspection += self.cost[0]
            obser_mark = 0.
            random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + self.observation_visual[new_hidden_state, j]
                if random_number <= obser_mark:
                    observation_value = j
                    break
            component_new_state[0: 5] = component_new_state[0: 5] * self.observation_visual[:,observation_value]
            component_new_state[0: 5] = component_new_state[0: 5] / np.sum(component_new_state[0: 5])

        if actions == 1 or actions == 3:
            cost_inspection += self.cost[1]
            obser_mark = 0.
            random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + self.observation_NDT[new_hidden_state, j]
                if random_number <= obser_mark:
                    observation_value = j
                    break
            component_new_state[0: 5] = component_new_state[0: 5] * self.observation_NDT[:, observation_value]
            component_new_state[0: 5] = component_new_state[0: 5] / np.sum(component_new_state[0: 5])

        """
        preventative maintenance action---------------------------------------------------------------------------------
        """
        cost_prevention = 0
        if actions == 2 or actions == 3:
            protection_time = self.protection[component_ID]
            cost_prevention += self.cost[2]

        """
        calculate the cost based on the cost & risk table---------------------------------------------------------------
        """
        risk = self.risk @ component_state
        reward = cost_inspection + cost_prevention + cost_repair + risk

        """
        calculate new state---------------------------------------------------------------------------------------------
        """
        ID = tf.one_hot(component_ID, 21).numpy()
        Time = tf.one_hot(time, 100).numpy()

        protection_time /= 20
        duration_time /= 30
        new_state = np.concatenate((component_new_state, [protection_time], duration_time, ID, Time))
        new_state = new_state.reshape(1, len(new_state))

        return new_state, reward, new_hidden_state

    def get_action(self, Actor_network, state, greedy=True):
        logit = Actor_network(state)
        prob = tf.nn.softmax(logit).numpy()
        if greedy:
            return np.argmax(prob.ravel())
        action = np.random.choice(logit.shape[1], p=prob.ravel())
        return action

    def future_estimation(self, initial_state, Actor_nn):
        # estimate whether performance become better
        # initial the parameters
        max_over_step = 20

        states_prediction = np.zeros((max_over_step, self.input_shape))
        action_prediction = np.zeros((max_over_step))
        reward_prediction = np.zeros((max_over_step))
        hidden_state_prediction = np.zeros((max_over_step))

        t = 0
        states = initial_state.copy()
        state_probs = initial_state[0, 0:5].flatten()
        prob = state_probs / np.sum(state_probs)
        hidden_state = np.random.choice(len(state_probs), p=prob)

        while t < max_over_step:
            hidden_state_prediction[t] = hidden_state
            action = self.get_action(Actor_nn, states, greedy=True)

            new_state, reward, hidden_state = self.step(states, action, hidden_state)
            states_prediction[t, :] = states.copy()
            action_prediction[t] = action
            reward_prediction[t] = reward

            states = new_state.copy()

            t += 1

        return states_prediction, action_prediction, reward_prediction, hidden_state_prediction


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # 启用高DPI缩放
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # 使用高DPI图标

    # 创建应用实例
    app = QApplication(sys.argv)

    # ===== Windows特有的应用ID设置（任务栏图标的关键）=====
    if sys.platform == 'win32':
        try:
            import ctypes

            # 设置一个唯一的应用ID（格式：公司名.应用名.版本号）
            app_id = 'Bridge management system.1.0.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            print(f"Windows app ID set successfully: {app_id}")
        except Exception as e:
            print(f"Windows app ID set successfully: {e}")

    # ===== 图标设置 =====
    # 使用绝对路径确保文件位置正确
    icon_path = 'logo.ico'

    # 转换为绝对路径
    icon_path = os.path.abspath(icon_path)

    # 创建QIcon对象（只创建一次）
    app_icon = QIcon(icon_path)

    # 设置应用程序图标（影响任务栏）
    app.setWindowIcon(app_icon)  # 注意：这里不需要再包装成QIcon

    # 初始化网络和环境
    Actor_network = ActorWithAttention(131, 5)
    Critic_network = CriticWithAttention(131)
    Environment = environment()

    # 创建主窗口
    ex = ActorCriticApp(Actor_network, Critic_network, Environment)

    # 设置窗口图标（影响标题栏）
    ex.setWindowIcon(app_icon)  # 使用同一个QIcon对象

    # 显示窗口
    ex.show()

    # 运行应用
    sys.exit(app.exec_())


