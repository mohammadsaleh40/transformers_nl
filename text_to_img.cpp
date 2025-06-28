#include <ft2build.h>
#include FT_FREETYPE_H

#include <harfbuzz/hb.h>
#include <harfbuzz/hb-ft.h>

#include <string>
#include <algorithm>
#include <cstdint>
#include <iostream>

// برای ذخیره‌سازی PNG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void draw_text(const std::string& text, const std::string& font_path, uint8_t image[64][128]) {
    FT_Library ft;
    FT_Init_FreeType(&ft);

    FT_Face face;
    FT_New_Face(ft, font_path.c_str(), 0, &face);
    FT_Set_Pixel_Sizes(face, 0, 21);

    hb_buffer_t *buf = hb_buffer_create();
    hb_buffer_add_utf8(buf, text.c_str(), -1, 0, -1);
    hb_buffer_set_direction(buf, HB_DIRECTION_RTL);
    hb_buffer_set_script(buf, HB_SCRIPT_ARABIC);
    hb_buffer_set_language(buf, hb_language_from_string("fa", -1));

    hb_shape(hb_ft_font_create(face, nullptr), buf, nullptr, 0);

    unsigned int count;
    hb_glyph_info_t *info = hb_buffer_get_glyph_infos(buf, &count);
    hb_glyph_position_t *pos = hb_buffer_get_glyph_positions(buf, &count);

    int x = 4, y = 21;
    for (unsigned int i = 0; i < count; ++i) {
        FT_Load_Glyph(face, info[i].codepoint, FT_LOAD_RENDER);
        FT_Bitmap bmp = face->glyph->bitmap;

        int glyph_x = x + (pos[i].x_offset >> 6);
        int glyph_y = y - face->glyph->bitmap_top;

        // محاسبه Bounding Box برای هر گلیف
        int bb_left = glyph_x;
        int bb_top = glyph_y;
        int bb_right = glyph_x + bmp.width;
        int bb_bottom = glyph_y + bmp.rows;

        std::cout << "Bounding Box گلیف [" << i << "]: "
                << "Left=" << bb_left << ", Top=" << bb_top
                << ", Right=" << bb_right << ", Bottom=" << bb_bottom << std::endl;

        // رندرینگ گلیف (مانند قبل)
        for (int row = 0; row < bmp.rows; ++row) {
            for (int col = 0; col < bmp.width; ++col) {
                int px = glyph_x + col;
                int py = glyph_y + row;
                if (px >= 0 && px < 128 && py >= 0 && py < 64) {
                    image[py][px] = std::min(image[py][px], static_cast<uint8_t>(255 - bmp.buffer[row * bmp.pitch + col]));
                }
            }
        }
        x += (pos[i].x_advance >> 6);
    }

    hb_buffer_destroy(buf);
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
}

int main() {
    uint8_t image[64][128];
    std::fill(&image[0][0], &image[0][0] + 64 * 128, 255);
    std::string font_path = "/usr/share/fonts/truetype/Vazirmatn-ExtraBold.ttf";
    std::string word = "سلام ببببب بریم به هیأت";
    draw_text(word, font_path, image);
    std::string filename = word + ".png";

    // تغییر نام فایل با استفاده از .c_str()
    stbi_write_png(filename.c_str(), 128, 64, 1, image, 128);

    // چاپ نام فایل به صورت دینامیک
    std::cout << "تصویر ذخیره شد: " << filename << std::endl;
    return 0;
}
