package drones;

import org.lwjgl.opengl.GL33;
import org.lwjgl.system.MemoryUtil;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.joml.Matrix4f;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;




abstract public class RenderableObject {
    protected static final List<String> FILES = Arrays.asList(
        "./src/drones/objects/cube.obj",
        "./src/drones/objects/model.obj",
        "./src/drones/objects/sphere.obj",
        "./src/drones/objects/flag.obj"
    );

    protected int vao;
    protected int vbo;
    protected int quadVao;
    protected boolean wasRenderCreated;
    protected Vector3f color;
    protected int usesTexture;
    protected Matrix4f transformation;
    protected float[] vertices;
    protected static int int32_False = 0;
    protected static int int32_True = 1;
    protected static FrameBuffer FB;
    private Texture texture;
    private int TEXTURE_UNIT;
    public int metallicLighting;

    
    

    public RenderableObject(ObjectParams params) {
        this.vertices = ObjectLoader.getObjectData(params.filename, params.hasNormal, params.hasTexture);
        this.color = new Vector3f(1, 0, 0);
        this.usesTexture = int32_False;
        this.metallicLighting = int32_False;
        createBuffers();
    }
    
    public void debug() {
    	for(int i = 0; i < Math.min(24, vertices.length); i += 8) {
    	    System.out.println("Vertex " + (i/8) + ": " + 
    	        vertices[i] + ", " + vertices[i+1] + ", " + vertices[i+2]);
    	}
    }
    
    abstract public Vector3f getPos();

    protected void createBuffers() {
        vao = GL33.glGenVertexArrays();
        GL33.glBindVertexArray(vao);

        vbo = GL33.glGenBuffers();
        GL33.glBindBuffer(GL33.GL_ARRAY_BUFFER, vbo);
        FloatBuffer vertexBuffer = MemoryUtil.memAllocFloat(vertices.length);
        vertexBuffer.put(vertices).flip();
        GL33.glBufferData(GL33.GL_ARRAY_BUFFER, vertexBuffer, GL33.GL_STATIC_DRAW);
        MemoryUtil.memFree(vertexBuffer);
    }
    public void setTexture(Texture texture, int texUnit) {
    	this.usesTexture = int32_True;
    	this.TEXTURE_UNIT = texUnit;
    	this.texture = texture;
    }

    public void createRenderable() {
        GL33.glBindVertexArray(vao);
        GL33.glBindBuffer(GL33.GL_ARRAY_BUFFER, vbo);

        GL33.glVertexAttribPointer(0, 3, GL33.GL_FLOAT, false, 8 * Float.BYTES, 0);
        GL33.glEnableVertexAttribArray(0);

        GL33.glVertexAttribPointer(1, 3, GL33.GL_FLOAT, false, 8 * Float.BYTES, 3 * Float.BYTES);
        GL33.glEnableVertexAttribArray(1);

        GL33.glVertexAttribPointer(2, 2, GL33.GL_FLOAT, false, 8 * Float.BYTES, 6 * Float.BYTES);
        GL33.glEnableVertexAttribArray(2);
        
        wasRenderCreated = true;
        
    }
    
    public void renderDepth(int depthProgram, Matrix4f perspective, Matrix4f view) {
    	if (!wasRenderCreated) {
            throw new IllegalStateException("Must create renderable first");
        }
        GL33.glEnable(GL33.GL_DEPTH_TEST);

        GL33.glUseProgram(depthProgram);
        
        FloatBuffer matrixBuffer = MemoryUtil.memAllocFloat(16);
        int modelLocation = GL33.glGetUniformLocation(depthProgram, "model");
        transformation.get(matrixBuffer);
        GL33.glUniformMatrix4fv(modelLocation, false, matrixBuffer);
        
        int perspLocation = GL33.glGetUniformLocation(depthProgram, "perspective");
        perspective.get(matrixBuffer);
        GL33.glUniformMatrix4fv(perspLocation, false, matrixBuffer);
        
        int viewLocation = GL33.glGetUniformLocation(depthProgram, "view");
        view.get(matrixBuffer);
        GL33.glUniformMatrix4fv(viewLocation, false, matrixBuffer);
        
        
        GL33.glBindVertexArray(vao);
        GL33.glDrawArrays(GL33.GL_TRIANGLES, 0, vertices.length / 8);
        
        MemoryUtil.memFree(matrixBuffer);
        int error;
        while ((error = GL33.glGetError()) != GL33.GL_NO_ERROR) {
            System.err.println("OpenGL Error: " + error);
        }
        
        
    }
    


    public void render(int shaderProgram, Vector3f eye, Vector4f light, Matrix4f view, Matrix4f perspective, Matrix4f lightViewMatrix) {
        if (!wasRenderCreated) {
            throw new IllegalStateException("Must create renderable first");
        }
        GL33.glEnable(GL33.GL_DEPTH_TEST);

        GL33.glUseProgram(shaderProgram);
        
        if (usesTexture == int32_True) {
        	texture.bind(TEXTURE_UNIT);
        	int mapLocation = GL33.glGetUniformLocation(shaderProgram, "map");
            GL33.glUniform1i(mapLocation, TEXTURE_UNIT);
        }
        
        FloatBuffer matrixBuffer = MemoryUtil.memAllocFloat(16);
        
        int lightProjectionLocation = GL33.glGetUniformLocation(shaderProgram, "lightProjectionMatrix");
        perspective.get(matrixBuffer);
        GL33.glUniformMatrix4fv(lightProjectionLocation, false, matrixBuffer);
        
        int lightViewMatrixLocation = GL33.glGetUniformLocation(shaderProgram, "lightViewMatrix");
        lightViewMatrix.get(matrixBuffer);
        GL33.glUniformMatrix4fv(lightViewMatrixLocation, false, matrixBuffer);
        
        

        int eyeLocation = GL33.glGetUniformLocation(shaderProgram, "eye");
        GL33.glUniform3f(eyeLocation, eye.x, eye.y, eye.z);

        int lightLocation = GL33.glGetUniformLocation(shaderProgram, "light");
        GL33.glUniform4f(lightLocation, light.x, light.y, light.z, light.w);

        int viewLocation = GL33.glGetUniformLocation(shaderProgram, "view");
        view.get(matrixBuffer);
        GL33.glUniformMatrix4fv(viewLocation, false, matrixBuffer);

        int perspLocation = GL33.glGetUniformLocation(shaderProgram, "perspective");
        perspective.get(matrixBuffer);
        GL33.glUniformMatrix4fv(perspLocation, false, matrixBuffer);

        int colorLocation = GL33.glGetUniformLocation(shaderProgram, "customColor");
        GL33.glUniform3f(colorLocation, color.x, color.y, color.z);

        int metalLocation = GL33.glGetUniformLocation(shaderProgram, "metal");
        GL33.glUniform1i(metalLocation, this.metallicLighting);
        
        int hasTexture = GL33.glGetUniformLocation(shaderProgram, "hasTexture");
        GL33.glUniform1i(hasTexture, this.usesTexture);

        int modelLocation = GL33.glGetUniformLocation(shaderProgram, "model");
        transformation.get(matrixBuffer);
        GL33.glUniformMatrix4fv(modelLocation, false, matrixBuffer);
        
        FB.bindTexture();
        int shadowMapLocation = GL33.glGetUniformLocation(shaderProgram, "depthMap");
        GL33.glUniform1i(shadowMapLocation, FB.TEXTURE_UNIT);

        GL33.glBindVertexArray(vao);
        GL33.glDrawArrays(GL33.GL_TRIANGLES, 0, vertices.length / 8);

        MemoryUtil.memFree(matrixBuffer);
        int error;
        while ((error = GL33.glGetError()) != GL33.GL_NO_ERROR) {
            System.err.println("OpenGL PError: " + error);
        }
        
    }

    public void cleanup() {
        GL33.glDeleteBuffers(vbo);
        GL33.glDeleteVertexArrays(vao);
        if (quadVao != 0) GL33.glDeleteVertexArrays(quadVao);
    }
}





class ObjectParams {
    public final String filename;
    public final boolean hasNormal;
    public final boolean hasTexture;

    public ObjectParams(String filename, boolean hasNormal, boolean hasTexture) {
        this.filename = filename;
        this.hasNormal = hasNormal;
        this.hasTexture = hasTexture;
    }
}