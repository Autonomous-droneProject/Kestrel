package drones;

import org.joml.Matrix4f;
import org.joml.Vector3f;

public class Sphere extends RenderableObject {
    private Vector3f scaleFactor;
    private Vector3f position;
    public int scale;
    
    public Sphere(int scale, Vector3f color, Vector3f position) {
        super(new ObjectParams(FILES.get(2), true, true));
        this.scaleFactor = new Vector3f(scale);
        this.scale = scale;
        this.color = new Vector3f(color);
        this.position = position;
        updateTransformation(position);
    }

    public void move(Vector3f position) {
    	this.position = position;
        updateTransformation(position);
    }
    
    public Vector3f getPos() {
    	return this.position;
    }
    
    

    private void updateTransformation(Vector3f position) {
    	this.position = position;
        transformation = new Matrix4f()
            .translate(position)
            .scale(scaleFactor);
    }
}
