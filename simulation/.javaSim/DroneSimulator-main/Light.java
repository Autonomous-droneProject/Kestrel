package drones;

import org.joml.Matrix4f;
import org.joml.Vector3f;
import java.lang.Math;



public class Light {
	private float lightDistance;
    private Vector3f lightDir;
    private Vector3f lightStartDirection;
    private Vector3f lightTarget;
    private Vector3f lightOrbitAxis;
    
    private static final int FOV = 60;
    private static final int NEAR = 1;
    private static final int FAR = 1000000;
    
    public Light() {
        lightDistance = 1000.0f;
        lightDir = new Vector3f(-0.0001f, 1, 0.0001f);
        lightStartDirection = new Vector3f(lightDir).normalize();
        lightTarget = new Vector3f(0.0f, 0.0f, 0.0f);
        lightOrbitAxis = new Vector3f(0.0f, 1.0f, 0.0f);
    }

    public void recomputeVariables() {
        lightStartDirection = new Vector3f(lightDir).normalize();
    }

    public Matrix4f getCameraMatrixLight(Vector3f lightPosition) {
        return new Matrix4f().lookAt(lightPosition, lightTarget, lightOrbitAxis);
    }
    
    public static Matrix4f getProjectionMatrix(float aspectRatio) {
    	//aspectRatio = (float)2000/800;
    	Matrix4f lightProjectionMatrix = new Matrix4f()
	            .perspective((float) Math.toRadians(FOV), aspectRatio, NEAR, FAR);
    	return lightProjectionMatrix;
    }
    

    public Vector3f[] getLightVector(float angle) {
        Vector3f lightVector = new Vector3f(lightStartDirection)
            .rotateAxis((float) Math.toRadians(angle), lightOrbitAxis.x, lightOrbitAxis.y, lightOrbitAxis.z);

        Vector3f lightPosition = new Vector3f(lightVector)
            .mul(lightDistance)
            .add(lightTarget);

        return new Vector3f[]{lightVector, lightPosition};
    }
    

}
