package drones;

import java.util.ArrayList;
import java.util.List;

public class DroneSimulator implements Scene {
	private List<RenderableObject> renderableStack;
	private Drone drone;
	
	
	DroneSimulator() {
		renderableStack = new ArrayList<>();
		drone = new Drone();
		renderableStack.add(drone);
	}
	@Override
	public Drone getDrone() {
		return drone;
	}

	@Override
	public List<RenderableObject> getRenderableStack() {
		return renderableStack;
	}

	@Override
	public void animate() {
		drone.animate();
		System.out.println(drone.getPos());
	}

	@Override
	public void pushToStack(RenderableObject object) {
		if (!object.wasRenderCreated) {
            object.createRenderable();
        }
        this.renderableStack.add(object);
		
	}

}
