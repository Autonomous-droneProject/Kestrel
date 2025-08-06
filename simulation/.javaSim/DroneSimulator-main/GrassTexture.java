package drones;

import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL33.*;
import org.lwjgl.stb.STBImage;

import java.nio.ByteBuffer;

public class GrassTexture implements Texture {
    private int textureId;
    private int samplerId;
    public static final int defaultTextureUnit = 2;

    public GrassTexture(String texturePath) {
       
        int[] width = new int[1];
        int[] height = new int[1];
        int[] channels = new int[1];

       
        ByteBuffer imgData = STBImage.stbi_load(texturePath, width, height, channels, 3);
        if (imgData == null) {
            throw new RuntimeException("Failed to load texture image: " + texturePath);
        }

        textureId = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, textureId);
       
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width[0], height[0], 0, GL_RGB, GL_UNSIGNED_BYTE, imgData);

        glGenerateMipmap(GL_TEXTURE_2D);
        
        samplerId = glGenSamplers();
        glSamplerParameteri(samplerId, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glSamplerParameteri(samplerId, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        
        STBImage.stbi_image_free(imgData);
    }

    public int getTextureId() {
        return textureId;
    }

    public int getSamplerId() {
        return samplerId;
    }

    public void bind(int textureUnit) {
        
        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glBindSampler(textureUnit, samplerId);
    }

    public void cleanup() {
        
        glDeleteTextures(textureId);
        glDeleteSamplers(samplerId);
    }
}
