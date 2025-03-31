#version 330 core
in vec3 in_position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;
void main() {
    gl_Position =  perspective * view * model * vec4(in_position, 1.0);
}
