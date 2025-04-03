package drones;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL30.*;

public class QuadRenderer {
	static float[] quadVertices = {
		    // Positions   // TexCoords
		    -1.0f,  1.0f,  0.0f, 1.0f, // Top-left
		    -1.0f, -1.0f,  0.0f, 0.0f, // Bottom-left
		     1.0f, -1.0f,  1.0f, 0.0f, // Bottom-right
		     1.0f,  1.0f,  1.0f, 1.0f  // Top-right
		};
	static int[] indices = {
		0, 1, 2,  2, 3, 0
	};
	private int quadVAO;
	private int quadVBO;
	private int quadEBO;
	
	private int quadShaderProgram;
		
	public QuadRenderer(int quadShaderProgram) {
		this.quadShaderProgram = quadShaderProgram;
		quadVAO = glGenVertexArrays();
		quadVBO = glGenBuffers();
		quadEBO = glGenBuffers();
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, quadVertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, false, 4 * Float.BYTES, 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 2, GL_FLOAT, false, 4 * Float.BYTES, 2 * Float.BYTES);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

	}
	public void render() {
		glUseProgram(quadShaderProgram);
		glBindVertexArray(quadVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	}
}
