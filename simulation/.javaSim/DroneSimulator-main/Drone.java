package drones;

import org.joml.Matrix3f;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.glfw.GLFW;
import org.joml.Math;

public class Drone extends RenderableObject {
    private float scale = 2.3f;
    protected Vector3f velocity;
    private Vector3f acceleration;
    protected Vector3f thrusts;
    protected Vector3f gravity = new Vector3f(0, -1.2f, 0);
    protected float damping = 0.0098f;
    private Vector3f position; 
    private static final int GROUND = 2;
    
    private float pitch = 0; // Tilt forward/back // Z
    private float roll = 0;  // Tilt left/right // X
    private float yaw = 0;   // Rotation left/right
    
    public Drone() {
        super(new ObjectParams(FILES.get(1), true, true));
        this.color = new Vector3f(0.255f, 0.145f, 0.612f);  
        this.position = new Vector3f(0,5,0);
        updateTransformation();
        this.velocity = new Vector3f(0,0,0);
        this.acceleration = new Vector3f(0,0,0);
        this.thrusts = new Vector3f(0,0,0);
        this.metallicLighting = int32_True;
    }

    public void setPos(float x, float y, float z) {
        this.position.x = x;
        this.position.y = y;
        this.position.z = z;
        updateTransformation();
    }
   
 
    public void rotate(float pitchChange, float rollChange, float yawChange) {
        this.pitch += pitchChange; 
        this.roll += rollChange;
        this.yaw += yawChange;
        updateTransformation();
    }
    
    public void reset() {
    	this.pitch = 0;
        this.roll = 0;
        this.yaw = 0;
    }
    
    public void move(float dir, String axis) {
        switch (axis.toLowerCase()) {
            case "x" -> this.position.x += dir;
            case "y" -> this.position.y += dir;
            case "z" -> this.position.z += dir;
        }
        updateTransformation();
    }

    private void updateTransformation() {
        transformation = new Matrix4f()
            .translate(this.position)
            .rotateY((float) Math.toRadians(yaw))
            .rotateX((float) Math.toRadians(pitch))
            .rotateZ((float) Math.toRadians(roll))
            .scale(scale);
    }
    
    public Vector3f getPos() {
    	return this.position;
    }
    public void applyPhysics(Vector3f thrusts, float dT) {
    	Matrix3f rotation = new Matrix3f()
                .rotateY((float) Math.toRadians(yaw))
                .rotateX((float) Math.toRadians(pitch))
                .rotateZ((float) Math.toRadians(roll));
    	
    
    	Vector3f thrustForce = new Vector3f(thrusts);
        rotation.transform(thrustForce);
        Vector3f acceleration = new Vector3f();
        acceleration.add(thrustForce);
        acceleration.add(gravity);
        velocity.fma(dT, acceleration); 
        velocity.mul(0.99f);
        
    	
    }
    
    private void checkForCrash() {
    	if (position.y < GROUND) {
        	position.y = GROUND;
        	velocity.x = 0;
        	velocity.y = 0;
        	velocity.z = 0;
        	reset();
        }
    	
    }
    
    private void checkForlostDrone() {
    	if (Math.abs(position.x) > 1000 || Math.abs(position.z) > 1000) {
        	position.x = 0;
        	position.z = 0;
        	velocity.x = 0;
        	velocity.y = 0;
        	velocity.z = 0;
        	reset();
        }
    }
    
    
    
    
    
    
    public void animate() {
    	
    	this.acceleration.set(thrusts);
    	thrusts.y = Math.min(thrusts.y, 1.3f);
    	applyPhysics(thrusts, 0.001f);
    	
        position = position.add(velocity);
        checkForCrash();
        checkForlostDrone();
        
    	updateTransformation();
    	
    }
    

}
