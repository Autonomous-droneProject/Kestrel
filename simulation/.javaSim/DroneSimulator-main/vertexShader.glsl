
#version 330 core

layout (location = 0) in vec4 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 model;
uniform mat4 lightViewMatrix;
uniform mat4 view, perspective;
uniform mat4 lightProjectionMatrix;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;
out vec4 f_PosLightSpace;

void main() {
    f_position = (model*in_position).xyz;
    f_normal = normalize(mat3(transpose(inverse(model)))*in_normal);
    f_uv = in_uv;
    f_PosLightSpace = lightProjectionMatrix * lightViewMatrix * vec4(f_position,1.0);
    gl_Position = perspective*view*vec4(f_position,1.);
}
