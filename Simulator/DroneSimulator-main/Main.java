package drones;

import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;

import java.nio.*;

import static org.lwjgl.glfw.Callbacks.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.opengl.GL30.*;





public class Main {
	
	private long window;
    private int width = 1000;
    private int height = 800;
    private float aspectRatio;
    
    private Floor floor;
    private GrassTexture grassTexture;
    private QuadRenderer QR;
    private Scene scene;
   
    private int mainShaderProgram;
    private int depthShaderProgram;
    private int quadShaderProgram;
    
    Matrix4f perspectiveMatrix;
	Matrix4f lightProjectionMatrix;

    private FrameBuffer FB;

    private Camera camera;
    private Light light;
    
    final private String[] vertex_shader_paths = {"./src/drones/vertexShader.glsl", "./src/drones/quadCodeVertex.glsl", "./src/drones/depthShader.glsl"};
    final private String[] fragment_shader_paths = {"./src/drones/fragmentShader.glsl", "./src/drones/quadCodeFragment.glsl", "./src/drones/depthShaderFragment.glsl"};
    final private String[] texturePaths = {"./src/drones/tex.png"};
    
    private float angle = 0f;
    private Matrix4f viewMatrix;
    private Vector3f eyePoint;
    private boolean pointSourceFlag;
   
    private String mode = "DroneSimulator"; // "DroneSimulator " , ObstacleAvoidance
    
    
    private void compileShaders() {
    	try {
    		mainShaderProgram = ShaderUtils.createProgramFromFiles(vertex_shader_paths[0], fragment_shader_paths[0]);
    		quadShaderProgram = ShaderUtils.createProgramFromFiles(vertex_shader_paths[1], fragment_shader_paths[1]);
    		depthShaderProgram = ShaderUtils.createProgramFromFiles(vertex_shader_paths[2], fragment_shader_paths[2]);
    	} catch (java.io.IOException e) {
    		System.err.println("ERROR OCCURRED " + e.toString());
    	}
        
    }
    
    private void setup_renderables() {
    	floor.createRenderable();	
    	for (RenderableObject o: scene.getRenderableStack()) {
    		o.createRenderable();
    	}
    }
    
    
	public void run() {
		init();
		if (mode.equals("DroneSimulator")) {
			scene = new DroneSimulator();
		} else if (mode.equals("ObstacleAvoidance")){
			scene = new ObstacleAvoidance();
		}
		ShaderUtils.checkGLError("after init");
		
		floor = new Floor();
        camera = new Camera();
        
        light = new Light();
        grassTexture = new GrassTexture(texturePaths[0]);
        floor.setTexture(grassTexture, GrassTexture.defaultTextureUnit);
        setup_renderables(); 
        FB = new FrameBuffer();
        compileShaders();
        QR = new QuadRenderer(quadShaderProgram);
       
		loop();
		glfwFreeCallbacks(window);
		glfwDestroyWindow(window);

		glfwTerminate();
		glfwSetErrorCallback(null).free();
	}
	
		
	private void init() {
		
		GLFWErrorCallback.createPrint(System.err).set();

		if ( !glfwInit() )
			throw new IllegalStateException("Unable to initialize GLFW");

		glfwDefaultWindowHints(); 
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); 
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); 
		GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 4);
		GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 1);
		GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);
		GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_FORWARD_COMPAT, GLFW.GLFW_TRUE);

		
		window = glfwCreateWindow(width, height, "Game", NULL, NULL);
		if ( window == NULL )
			throw new RuntimeException("Failed to create the GLFW window");
		aspectRatio = (float) width / height;

		glfwSetFramebufferSizeCallback(window, (win, w, h) -> {
	        width = w;
	        height = h;
	        glViewport(0, 0, width, height);
	    });
		
		glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
			if ( key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE )
				glfwSetWindowShouldClose(window, true);
			UserInput.setCameraMovements(camera, window, action, key);
			UserInput.setCameraMovementRotationMovements(camera, window, action, key);	
			if (mode.equals("DroneSimulator")) {
				UserInput.setDroneMovements(scene.getDrone(), window, action, key);
			} else {
				UserInput.setAutonomousControls(scene.getDrone(), window, action, key);
			}
			
		});
		
		try ( MemoryStack stack = stackPush() ) {
			IntBuffer pWidth = stack.mallocInt(1); 
			IntBuffer pHeight = stack.mallocInt(1); 
			
			glfwGetWindowSize(window, pWidth, pHeight);
			GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			glfwSetWindowPos(
				window,
				(vidmode.width() - pWidth.get(0)) / 2,
				(vidmode.height() - pHeight.get(0)) / 2
			);
			width = pWidth.get(0);
			height = pHeight.get(0);
		} 
		aspectRatio = width / height;
		glfwMakeContextCurrent(window);
		glfwSwapInterval(0);	
		GL.createCapabilities();
		
	    glfwShowWindow(window);
	    System.out.println("Created Context");
	    
	    
		
	}
	
	
	private void renderShadowMap(Vector4f light, Vector3f eye, Matrix4f view, Matrix4f perspective) {
        glEnable(GL_DEPTH_TEST);
        glViewport(0,0, FrameBuffer.shadowMapSize, FrameBuffer.shadowMapSize);
        floor.renderDepth(depthShaderProgram, perspective, view);
        for (RenderableObject obj : scene.getRenderableStack()) {
            obj.renderDepth(depthShaderProgram, perspective, view);
        }
	}
	
	private void renderScene(Vector4f light, Vector3f eye, Matrix4f view, Matrix4f perspective, Matrix4f lightViewMatrix) {
        
		RenderableObject.FB = FB;
        for (RenderableObject obj : scene.getRenderableStack()) {
            obj.render(mainShaderProgram, eye, light, view, perspective, lightViewMatrix);
        }
        floor.render(mainShaderProgram, eye, light, view, perspective, lightViewMatrix);
        glBindTexture(GL_TEXTURE_2D, 0);
        
	}
	
	
	@SuppressWarnings("unused")
	private void renderDepth() {
		FB.bindTexture();
		ShaderUtils.setUniform(quadShaderProgram, "shadowMapSampler", FB.TEXTURE_UNIT);
		QR.render();
        glBindTexture(GL_TEXTURE_2D, 0);
        
	}
	
	protected void renderFrame() {
		
		Vector3f lightVector = new Vector3f();
        Vector3f lightPosition = new Vector3f();
        Vector3f[] light_details = light.getLightVector(angle);
        lightVector = light_details[0];
        lightPosition = light_details[1];
        
        Matrix4f lightViewMatrix = new Matrix4f();
        lightViewMatrix = light.getCameraMatrixLight(lightPosition);
        viewMatrix = camera.getCameraMatrix();
        eyePoint = camera.eyePoint;
        
        Vector4f lightVec = pointSourceFlag ? 
            new Vector4f(lightPosition, 1.0f) : 
            new Vector4f(lightVector, 0.0f);
        
        scene.animate();
        FB.bind();
        FB.clear(); 
        glClearDepth(1.0);
        glEnable(GL_DEPTH_TEST);
        
        renderShadowMap(lightVec, eyePoint, lightViewMatrix, lightProjectionMatrix); 
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0,0,width*2,height*2);
        angle += 0.001f;
        renderScene(lightVec, eyePoint, viewMatrix, perspectiveMatrix, lightViewMatrix); 
		
	}
	
	
	private void loop() {
		
		
		perspectiveMatrix = camera.getPerspectiveMatrix(aspectRatio);
		lightProjectionMatrix = Light.getProjectionMatrix(aspectRatio);
		pointSourceFlag = true;
		viewMatrix = camera.getCameraMatrix();
		eyePoint = camera.eyePoint;
		
		while ( !glfwWindowShouldClose(window) ) {
			glfwPollEvents();
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			renderFrame();
            glfwSwapBuffers(window);
            
		}
		scene.cleanup();
		
		
	}
	

	public static void main(String[] args) {
		new Main().run();

	}

}
