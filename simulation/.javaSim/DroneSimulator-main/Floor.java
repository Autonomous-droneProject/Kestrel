package drones;

import org.joml.Matrix4f;
import org.joml.Vector3f;

public class Floor extends RenderableObject {
	
	private Vector3f position;
	
    public Floor() {
        super(new ObjectParams(FILES.get(0), true, true));
        this.color = new Vector3f(0.11f, 0.62f, 0.13f);  // 28, 158, 33 normalized
        //this.usesTexture = true;
        position = new Vector3f(0,-1,0);
        updateTransformation();
       
    }
    
    public Vector3f getPos() {
    	return position;
    }

    private void updateTransformation() {
        transformation = new Matrix4f()
            .translate(this.position)
            .scale(500, 1, 500);
    }
    
   
}
