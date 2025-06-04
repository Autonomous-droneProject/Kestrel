package drones;

import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

import org.joml.Vector3f;

public class UserInput {
	long window;
	static final float SPEED = 0.1f;
	static final float MAX_SPEED = 2f;
	static Vector3f acceleration = new Vector3f(0,0,0);
	
    
	public UserInput(long window, int key, int scancode, int anction, int mods) {
		this.window = window;
	}
	
	public static void setDroneMovements(Drone drone, long window, int action, int key) {
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_R && action != GLFW_RELEASE) {
			drone.setPos(0,2,0);
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_U && action != GLFW_RELEASE) {
			drone.thrusts.y += SPEED;
	
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_U && action == GLFW_RELEASE){
			drone.thrusts.y = 0; 
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_I && action != GLFW_RELEASE) {
			drone.rotate(-10,0,0); // Z
	
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_I && action == GLFW_RELEASE){
			drone.rotate(0,0,0);
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_K && action != GLFW_RELEASE) {
			drone.rotate(10,0,0);
			
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_K && action == GLFW_RELEASE) {
			drone.rotate(0,0,0);
		}

		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_J && action != GLFW_RELEASE) {
			drone.rotate(0,10,0); // X
			
		} else if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_J && action == GLFW_RELEASE){
			drone.rotate(0,0,0);
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_L && action != GLFW_RELEASE) {
			drone.rotate(0,-10,0);
			
			
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_L && action == GLFW_RELEASE){
			drone.rotate(0,0,0);
		}
		
		if (drone.velocity.x > MAX_SPEED) {
			drone.velocity.x = MAX_SPEED;
		}
		if (drone.velocity.z > MAX_SPEED) {
			drone.velocity.z = MAX_SPEED;
		}
	}
	
	public static void setAutonomousControls(Drone drone, long window, int action, int key) {
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_R && action != GLFW_RELEASE) {
			drone.setPos(0,2,0);
		}
		
	}
	
	
	public static void setCameraMovements(Camera camera, long window, int action, int key) {
		
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_W && action != GLFW_RELEASE) {
			camera.velocity.z += SPEED;
	
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_W && action == GLFW_RELEASE){
			camera.velocity.z = 0; 
		}	
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_S && action != GLFW_RELEASE) {
			camera.velocity.z -= SPEED;
			
			
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_S && action == GLFW_RELEASE) {
			camera.velocity.z = 0; 
		}
			
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_A && action != GLFW_RELEASE) {
			camera.velocity.x += SPEED;
			
			
		} else if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_A && action == GLFW_RELEASE){
			camera.velocity.x = 0;
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_D && action != GLFW_RELEASE) {
			camera.velocity.x -= SPEED;
			
			
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_D && action == GLFW_RELEASE){
			camera.velocity.x = 0;
		}
		
		if (camera.velocity.x > MAX_SPEED) {
			camera.velocity.x = MAX_SPEED;
		}
		if (camera.velocity.z > MAX_SPEED) {
			camera.velocity.z = MAX_SPEED;
		}
			
		
	}
	
	public static void setCameraMovementRotationMovements(Camera camera, long window, int action, int key) {
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_UP && action != GLFW_RELEASE) {
			camera.rotationVelocity.x -= SPEED;
			if (camera.deg_angles.x < -50) {
				camera.deg_angles.x = -50;
				camera.rotationVelocity.x = 0;
			}
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_UP && action == GLFW_RELEASE) {
			camera.rotationVelocity.x = 0;
		}
			
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_DOWN && action != GLFW_RELEASE) {
			camera.rotationVelocity.x += SPEED;
			if (camera.deg_angles.x > 20) {
				camera.deg_angles.x = 20;
				camera.rotationVelocity.x = 0;
			}	
			
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_DOWN && action == GLFW_RELEASE) {
			camera.rotationVelocity.x = 0;
		}
		
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_LEFT && action != GLFW_RELEASE) {
			camera.rotationVelocity.y += SPEED;
		} else if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_LEFT && action == GLFW_RELEASE) {
			camera.rotationVelocity.y = 0;
		}
			
		if ( key == org.lwjgl.glfw.GLFW.GLFW_KEY_RIGHT  && action != GLFW_RELEASE) {
			camera.rotationVelocity.y -= SPEED;
		} else if (key == org.lwjgl.glfw.GLFW.GLFW_KEY_RIGHT  && action == GLFW_RELEASE)
			camera.rotationVelocity.y = 0;
		
	}
	
	
}
