from PIL import Image

def convert_to_ico(jpg_path, ico_path, sizes=[(16,16), (32,32), (48,48)]):
    img = Image.open(jpg_path)
    img.save(ico_path, format='ICO', sizes=sizes)

convert_to_ico('logo.jpg', 'logo.ico')