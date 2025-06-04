package drones;

import static org.lwjgl.opengl.GL11.GL_DEPTH_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL30.GL_FRAMEBUFFER;
import static org.lwjgl.opengl.GL30.glBindFramebuffer;
import org.lwjgl.opengl.GL45;



import java.nio.ByteBuffer;

public class FrameBuffer {
    public int depthTexture;
    private int fbo;
    private int colorTexture;
    private int depthSampler;
    protected int TEXTURE_UNIT = 3;
    protected static final int shadowMapSize = 2048;

    public FrameBuffer() {
    	
    	createDepthTexture();
        
        fbo = GL45.glGenFramebuffers();
        GL45.glBindFramebuffer(GL45.GL_FRAMEBUFFER, fbo);
        GL45.glFramebufferTexture2D(GL45.GL_FRAMEBUFFER, GL45.GL_COLOR_ATTACHMENT0, GL45.GL_TEXTURE_2D, colorTexture, 0);
        GL45.glFramebufferTexture2D(GL45.GL_FRAMEBUFFER, GL45.GL_DEPTH_ATTACHMENT, GL45.GL_TEXTURE_2D, depthTexture, 0);

        if (GL45.glCheckFramebufferStatus(GL45.GL_FRAMEBUFFER) != GL45.GL_FRAMEBUFFER_COMPLETE) {
            throw new RuntimeException("Framebuffer is not complete");
        }

        depthSampler = GL45.glGenSamplers();
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_MIN_FILTER, GL45.GL_LINEAR);
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_MAG_FILTER, GL45.GL_LINEAR);
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_WRAP_S, GL45.GL_CLAMP_TO_EDGE);
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_WRAP_T, GL45.GL_CLAMP_TO_EDGE);
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_COMPARE_MODE, GL45.GL_COMPARE_REF_TO_TEXTURE);
        GL45.glSamplerParameteri(depthSampler, GL45.GL_TEXTURE_COMPARE_FUNC, GL45.GL_LESS);

        GL45.glBindSampler(0, depthSampler);
        
    }
    
    
    void createDepthTexture() {
    	depthTexture = GL45.glGenTextures();
        GL45.glBindTexture(GL45.GL_TEXTURE_2D, depthTexture);
        GL45.glTexImage2D(GL45.GL_TEXTURE_2D, 0, GL45.GL_DEPTH_COMPONENT32F, shadowMapSize, shadowMapSize, 0, GL45.GL_DEPTH_COMPONENT, GL45.GL_FLOAT, (ByteBuffer) null);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_COMPARE_MODE, GL45.GL_COMPARE_REF_TO_TEXTURE);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_COMPARE_FUNC, GL45.GL_LESS);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_MIN_FILTER, GL45.GL_NEAREST);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_MAG_FILTER, GL45.GL_NEAREST);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_WRAP_S, GL45.GL_CLAMP_TO_EDGE);
        GL45.glTexParameteri(GL45.GL_TEXTURE_2D, GL45.GL_TEXTURE_WRAP_T, GL45.GL_CLAMP_TO_EDGE);
        colorTexture = GL45.glGenTextures();
        GL45.glBindTexture(GL45.GL_TEXTURE_2D, colorTexture);
        GL45.glTexImage2D(GL45.GL_TEXTURE_2D, 0, GL45.GL_RGBA8, shadowMapSize, shadowMapSize, 0, GL45.GL_RGBA, GL45.GL_UNSIGNED_BYTE, (ByteBuffer) null);
        
        
    }
    
    void bindTexture() {
    	GL45.glActiveTexture(GL45.GL_TEXTURE3);
    	GL45.glBindTexture(GL45.GL_TEXTURE_2D, depthTexture);
    	GL45.glBindSampler(TEXTURE_UNIT, depthSampler);
        
       
    }
   
    void bind() {
    	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }
    
    void unbind() {
    	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void clear() {
    	glClear(GL_DEPTH_BUFFER_BIT);
    }
}
