import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import cv2
import numpy as np

# For Testing
image_path = "PartB_00000.jpg"
image_array = cv2.imread(image_path, 256)
image_array = cv2.resize(image_array, (100, 100))               # if needed
alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
rgba = np.concatenate([image_array, alpha], axis=2)     # HxWx4, RGBA uint8
tex = (rgba.astype(np.float32) / 255.0).ravel() # flat float32 in [0,1]

dpg.create_context()
dpg.create_viewport(title='Kestrel Tracking Management', width=1080, height=1920)

texture_data = image_array

# Test add texture
# for i in range(0, 100 * 100):
#     texture_data.append(255 / 255)
#     texture_data.append(0)
#     texture_data.append(255 / 255)
#     texture_data.append(255 / 255)

with dpg.texture_registry():
    texture_id = dpg.add_dynamic_texture(width=100, height=100, default_value=tex, tag="texture_tag") # Textures cannot be resized once created

def _update_dynamic_textures(sender, app_data, user_data):
    new_color = dpg.get_value(sender)
    new_color[0] = new_color[0] / 255
    new_color[1] = new_color[1] / 255
    new_color[2] = new_color[2] / 255
    new_color[3] = new_color[3] / 255

    new_texture_data = []
    for i in range(0, 100 * 100):
        new_texture_data.append(new_color[0])
        new_texture_data.append(new_color[1])
        new_texture_data.append(new_color[2])
        new_texture_data.append(new_color[3])

    dpg.set_value("texture_tag", new_texture_data)

with dpg.window(tag="Primary Window", label="Whitelist Tracks"):
    dpg.add_text("Select the bounding boxes you wish to track.")
    dpg.set_primary_window("Primary Window", True)
    with dpg.table(header_row=False):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()
        
        for i in range(0, 7):
            with dpg.table_row():
                for j in range(0, 6):
                    dpg.add_image("texture_tag")
        
        # dpg.add_color_picker((255, 0, 255, 255), label="Texture",
        #                  no_side_preview=True, alpha_bar=True, width=200,
        #                  callback=_update_dynamic_textures)


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()