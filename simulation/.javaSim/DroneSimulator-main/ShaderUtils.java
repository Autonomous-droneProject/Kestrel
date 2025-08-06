package drones;

import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL33;
import org.lwjgl.system.MemoryUtil;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.opengl.GL11;
import static org.lwjgl.opengl.GL11.GL_NO_ERROR;
import static org.lwjgl.opengl.GL11.glGetError;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class ShaderUtils {
    
    public static String readShaderFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        return Files.readString(path);
    }


    public static String readShaderFileWithBuffer(String filePath) throws IOException {
        StringBuilder shaderSource = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                shaderSource.append(line).append("\n");
            }
        }
        return shaderSource.toString();
    }

    
    public static int createProgramFromFiles(String vertexShaderPath, String fragmentShaderPath) throws IOException {
        String vertexShaderSource = readShaderFile(vertexShaderPath);
        String fragmentShaderSource = readShaderFile(fragmentShaderPath);
        return createProgram(vertexShaderSource, fragmentShaderSource);
    }
    
    public static void checkGLError(String label) {
	    int error;
	    while ((error = glGetError()) != GL_NO_ERROR) {
	        System.out.println("OpenGL Error " + error + " at " + label);
	    }
	}

    
    public static int createProgram(String vertexShaderSource, String fragmentShaderSource) {
        
    	int vertexShaderId = createShader(vertexShaderSource, GL20.GL_VERTEX_SHADER);
        int fragmentShaderId = createShader(fragmentShaderSource, GL20.GL_FRAGMENT_SHADER);
		

        int programId = GL20.glCreateProgram();        
        
        GL20.glAttachShader(programId, vertexShaderId);
        GL20.glAttachShader(programId, fragmentShaderId);
        GL20.glLinkProgram(programId);
        
        if (GL20.glGetProgrami(programId, GL20.GL_LINK_STATUS) == GL11.GL_FALSE) {
            String infoLog = GL20.glGetProgramInfoLog(programId);
            throw new RuntimeException("Shader program linking failed:\n" + infoLog);
        }
        
      
        GL20.glDeleteShader(vertexShaderId);
        GL20.glDeleteShader(fragmentShaderId);
        
        

        return programId;
    }

    
    private static int createShader(String shaderSource, int shaderType) {
        int shaderId = GL20.glCreateShader(shaderType);
        GL20.glShaderSource(shaderId, shaderSource);
        GL20.glCompileShader(shaderId);

        if (GL20.glGetShaderi(shaderId, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
            String infoLog = GL20.glGetShaderInfoLog(shaderId);
            String shaderTypeStr = (shaderType == GL20.GL_VERTEX_SHADER) ? "Vertex" : "Fragment";
            throw new RuntimeException(shaderTypeStr + " shader compilation failed:\n" + infoLog);
        }
        return shaderId;
    }

    public static void deleteProgram(int programId) {
        GL20.glDeleteProgram(programId);
    }

    public static int getUniformLocation(int programId, String uniformName) {
        int location = GL20.glGetUniformLocation(programId, uniformName);
        if (location == -1) {
            throw new RuntimeException("Could not find uniform variable '" + uniformName + "'");
        }
        return location;
    }

    
    public static int getAttributeLocation(int programId, String attributeName) {
        int location = GL20.glGetAttribLocation(programId, attributeName);
        if (location == -1) {
            throw new RuntimeException("Could not find attribute variable '" + attributeName + "'");
        }
        return location;
    }

	public static void setUniform(int mainShaderProgram, String string, Vector3f vec3Property) {
		GL33.glUseProgram(mainShaderProgram);
		int location = GL33.glGetUniformLocation(mainShaderProgram, string);
        GL33.glUniform3f(location, vec3Property.x, vec3Property.y, vec3Property.z);	
		
	}
	
	public static void setUniform(int mainShaderProgram, String string, Matrix4f vec4Property) {
		GL33.glUseProgram(mainShaderProgram);
		FloatBuffer matrixBuffer = MemoryUtil.memAllocFloat(16);
		int location = GL33.glGetUniformLocation(mainShaderProgram, string);
        vec4Property.get(matrixBuffer);
        GL33.glUniformMatrix4fv(location, false, matrixBuffer);
		
	}


	public static void setUniform(int mainShaderProgram, String string, int i) {
		GL33.glUseProgram(mainShaderProgram);
		int mapLocation = GL33.glGetUniformLocation(mainShaderProgram, string);
		if (mapLocation == -1) {
	        System.out.println("Warning: Uniform '" + mapLocation + "' not found in shader!");
	        return;
	    }
        GL33.glUniform1i(mapLocation, i);
		
	}


	public static void setUniform(int mainShaderProgram, String string, Vector4f vec4Property) {
		GL33.glUseProgram(mainShaderProgram);
		int location = GL33.glGetUniformLocation(mainShaderProgram, string);
        GL33.glUniform4f(location, vec4Property.x, vec4Property.y, vec4Property.z, vec4Property.w);	
		
		
	}
}