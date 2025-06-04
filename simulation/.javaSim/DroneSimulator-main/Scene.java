package drones;

import java.util.List;

public interface Scene {
	
	public List<RenderableObject> getRenderableStack();
	
	public Drone getDrone();
	
	public void animate();
	
	public void pushToStack(RenderableObject object);
	
	default void cleanup() {
		List<RenderableObject> objects = getRenderableStack();
		for (RenderableObject o: objects) {
			o.cleanup();
		}
	}
	
	
}

