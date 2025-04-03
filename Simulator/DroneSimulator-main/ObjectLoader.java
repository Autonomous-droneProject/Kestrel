package drones;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.joml.Vector2f;
import org.joml.Vector3f;

public class ObjectLoader {
    public static float[] getObjectData(String filename, boolean includeNormals, boolean includeTextures) {
        List<Vector3f> vertices = new ArrayList<>();
        List<Vector3f> normals = new ArrayList<>();
        List<Vector2f> textures = new ArrayList<>();
        List<Face> faces = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split("\\s+");
                if (tokens.length == 0) continue;

                switch (tokens[0]) {
                    case "v" -> { 
                        if (tokens.length >= 4) {
                            Vector3f vertex = new Vector3f(
                                Float.parseFloat(tokens[1]),
                                Float.parseFloat(tokens[2]),
                                Float.parseFloat(tokens[3])
                            );
                            vertices.add(vertex);
                        }
                    }
                    case "vt" -> { 
                        if (tokens.length >= 3) {
                            Vector2f tex = new Vector2f(
                                Float.parseFloat(tokens[1]),
                                Float.parseFloat(tokens[2])
                            );
                            textures.add(tex);
                        }
                    }
                    case "vn" -> { 
                        if (tokens.length >= 4) {
                            Vector3f normal = new Vector3f(
                                Float.parseFloat(tokens[1]),
                                Float.parseFloat(tokens[2]),
                                Float.parseFloat(tokens[3])
                            );
                            normals.add(normal);
                        }
                    }
                    case "f" -> { 
                        if (tokens.length >= 4) {
                            Face face = new Face();
                            for (int i = 1; i < tokens.length; i++) {
                                String[] parts = tokens[i].split("/");
                                FaceVertex fv = new FaceVertex();
                                
                                
                                fv.vertexIndex = Integer.parseInt(parts[0]) - 1;
                                
                                
                                if (parts.length > 1 && !parts[1].isEmpty()) {
                                    fv.textureIndex = Integer.parseInt(parts[1]) - 1;
                                }
                                
                                
                                if (parts.length > 2 && !parts[2].isEmpty()) {
                                    fv.normalIndex = Integer.parseInt(parts[2]) - 1;
                                }
                                
                                face.vertices.add(fv);
                            }
                            faces.add(face);
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load OBJ file: " + filename, e);
        }

        return convertToVertexArray(vertices, normals, textures, faces, includeNormals, includeTextures);
    }

    private static float[] convertToVertexArray(
            List<Vector3f> vertices,
            List<Vector3f> normals,
            List<Vector2f> textures,
            List<Face> faces,
            boolean includeNormals,
            boolean includeTextures) {
        
        List<Float> vertexArray = new ArrayList<>();

        for (Face face : faces) {
            for (int i = 1; i < face.vertices.size() - 1; i++) {
                processFaceVertex(face.vertices.get(0), vertices, normals, textures, 
                    includeNormals, includeTextures, vertexArray);
                processFaceVertex(face.vertices.get(i), vertices, normals, textures, 
                    includeNormals, includeTextures, vertexArray);
                processFaceVertex(face.vertices.get(i + 1), vertices, normals, textures, 
                    includeNormals, includeTextures, vertexArray);
            }
        }

        float[] result = new float[vertexArray.size()];
        for (int i = 0; i < vertexArray.size(); i++) {
            result[i] = vertexArray.get(i);
        }
        return result;
    }

    private static void processFaceVertex(
            FaceVertex fv,
            List<Vector3f> vertices,
            List<Vector3f> normals,
            List<Vector2f> textures,
            boolean includeNormals,
            boolean includeTextures,
            List<Float> output) {
        
        Vector3f vertex = vertices.get(fv.vertexIndex);
        output.add(vertex.x);
        output.add(vertex.y);
        output.add(vertex.z);

       
        if (includeNormals) {
            Vector3f normal = fv.normalIndex >= 0 ? 
                normals.get(fv.normalIndex) : 
                new Vector3f(0, 1, 0);  
            output.add(normal.x);
            output.add(normal.y);
            output.add(normal.z);
        }

        if (includeTextures) {
            Vector2f texture = fv.textureIndex >= 0 ? 
                textures.get(fv.textureIndex) : 
                new Vector2f(0, 0);  
            output.add(texture.x);
            output.add(texture.y);
        }
    }

    

    private static class FaceVertex {
        int vertexIndex = -1;
        int textureIndex = -1;
        int normalIndex = -1;
    }

    private static class Face {
        List<FaceVertex> vertices = new ArrayList<>();
    }
}
