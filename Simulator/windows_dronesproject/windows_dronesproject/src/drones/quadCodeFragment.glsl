#version 330 core

in vec2 uv;
uniform sampler2D shadowMapSampler;
layout (location = 0) out vec4 out_color;

void main() {
    float depthValue = texture(shadowMapSampler, uv).r;
    if (depthValue == 0.0) {
        out_color = vec4(1, 0, 0, 1); 
    } else {
        out_color = vec4(vec3(depthValue), 1.0);
    }
    //out_color = vec4(vec3(depthValue), 1.0);
}
