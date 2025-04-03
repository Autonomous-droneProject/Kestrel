#version 330 core

in vec3 f_position;
in vec3 f_normal;
in vec2 f_uv;
in vec4 f_PosLightSpace;

uniform vec3 eye;
uniform sampler2D map;
uniform vec4 light;
uniform vec3 customColor;
uniform sampler2D depthMap;


uniform bool metal;
uniform bool hasTexture;
const float shininess = 0.6;


layout (location = 0) out vec4 out_color;

void main() {
    vec3 N = normalize(f_normal);
    vec3 L = normalize(light.xyz);
    if (light.w > 0.) L = normalize(light.xyz-f_position);
    
    vec3 V = normalize(eye-f_position);
    vec3 H = normalize(L+V);
    
    //vec3 baseColor = vec3(28.0 / 255.0, 158.0 / 255.0, 33.0 / 255.0);
    vec3 baseColor = customColor;

    vec3 fragPos3D = f_PosLightSpace.xyz / f_PosLightSpace.w;
    float depthCurrent = (fragPos3D.z+1.0)/2.0;
    vec2 shadowCoord = f_PosLightSpace.xy / f_PosLightSpace.w * 0.5 + 0.5;
    vec2 uv = (fragPos3D.xy+1.0)/2.0;
    float depthStored = texture(depthMap, uv).r;
    float shadowFactor = 1.0;
    //float bias = max(0.05 * (1.0 - dot(N, L)), 0.005);
    float bias = 0.001;

    vec3 color  = ((metal)? pow(clamp(dot(H,N), 0., 1.), shininess):clamp(dot(N,L),0.,1.))*baseColor;
    

    if (depthStored+bias < depthCurrent) {
        shadowFactor = 0.8;
    }

    if (hasTexture) {
        baseColor = texture(map, f_uv).rgb;
        vec3 texcolor  = ((metal)? pow(clamp(dot(H,N), 0., 1.), shininess):clamp(dot(N,L),0.,1.))*baseColor;
        out_color = vec4(texcolor*shadowFactor,1.);
        
    } else {
        out_color = vec4(color*shadowFactor,1.);
    }

    
    
}
