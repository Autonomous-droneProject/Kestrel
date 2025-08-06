package drones;

import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import org.joml.Vector3f;
import java.util.Random;

public class ObstacleAvoidance implements Scene {
	private List<RenderableObject> renderableStack;
	private List<Sphere> obstacles;
	private Drone drone;
	private Flag flag;
	private static int NUMBER_OBJECTS = 20;
	private static int GROUND = 2;
	
	private void generateRandomObstacles() {
		
		Random random = new Random();
		
		for (int i=0;i<NUMBER_OBJECTS;i++) {
			int scale = random.nextInt(5) + 2;
			Vector3f position = new Vector3f((float)random.nextInt(100), (float) random.nextInt(20), (float) random.nextInt(100));
			Vector3f color = new Vector3f((float)Math.random(), (float)Math.random(), (float)Math.random());
			Sphere sph = new Sphere(scale, color, position);
			obstacles.add(sph);
			renderableStack.add(sph);
		}
		
	}
	
	ObstacleAvoidance() {
		renderableStack = new ArrayList<>();
		obstacles = new ArrayList<>();
		generateRandomObstacles();
		drone = new Drone();
		drone.setPos(-100, GROUND, -100);
		flag = new Flag();
		flag.setPos(150, 10, 150);
		renderableStack.add(drone);
		renderableStack.add(flag);
	}
	
	private Vector3f computeRepulsiveField() {
	    Vector3f current = drone.getPos();
	    float p0 = 50.0f; 
	    float epsilon = 0.0001f;
	    
	    Vector3f result = new Vector3f();
	    float coeff = 1000000.0f;

	    for (Sphere obstacle : obstacles) {
	        Vector3f position = obstacle.getPos();
	        Vector3f distance = new Vector3f(current).sub(position); 
	        float norm = distance.length() + epsilon; 
	        
	        if (norm < p0) {
	            float forceMagnitude = (1.0f / norm) - (1.0f / p0);
	            forceMagnitude *= (1.0f / (norm * norm)); 

	            Vector3f direction = new Vector3f(distance).normalize(); 
	            Vector3f repulsiveForce = new Vector3f(direction).mul(forceMagnitude * coeff * (obstacle.scale * obstacle.scale )); 
	            
	            result.add(repulsiveForce);
	        }
	    }
	    
	    if (current.y <= (GROUND)) {
	    	float gdist = Math.max(current.y, epsilon);
	    	float forceMagnitude = (1.0f / gdist) - (1.0f / GROUND);
	    	forceMagnitude *= (1.0f / (gdist * gdist));
	    	Vector3f groundRepulsion = new Vector3f(0, 1, 0).mul(-forceMagnitude * coeff);
	        result.add(groundRepulsion);
	    }

	    return result;
	}

	

	public Vector3f computeAttractiveField() {
        Vector3f current = drone.getPos();
        
        Vector3f distance = flag.getPos().sub(current);

        float norm = distance.length();
        float forceMagnitude = (float) (0.5 * norm * norm);

        Vector3f direction = new Vector3f(distance).div( (float) ((float)norm + 0.0001));

        return direction.mul(forceMagnitude);
    }
	
	public boolean didReachTarget() {
		Vector3f current = drone.getPos();
		Vector3f flagPos = flag.getPos();
		int minDistance = 10;
		if (Math.abs(flagPos.x - current.x) < minDistance && Math.abs(flagPos.z - current.z) < minDistance) {
			return true;
		}
		return false;
		
	}

	@Override
	public List<RenderableObject> getRenderableStack() {
		return renderableStack;
	}
	@Override 
	public void animate() {
		Vector3f rF = computeRepulsiveField();
		Vector3f movement = new Vector3f(computeAttractiveField()).add(rF);
		
		float alpha = 0.00000001f;
		drone.thrusts = new Vector3f(0, 10f * movement.y, 0);
		drone.rotate(alpha * movement.z, 0, 0);
		drone.rotate(0, -alpha * movement.x, 0);
		
		if (didReachTarget()) {
			drone.thrusts.y = 0;
		}
		drone.animate();
		
		
	}
	@Override
	public Drone getDrone() {
		return drone;
	}

	
	@Override
	public void pushToStack(RenderableObject object) {
        if (!object.wasRenderCreated) {
            object.createRenderable();
        }
        this.renderableStack.add(object);
    }

}
