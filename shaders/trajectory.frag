// DEEP OCEAN ROBOT VISION 4D - LED trajectory strip (GLSL MAT, pixel). Vectors: uColor(vec3 neon colour), uGain, uOpacity
uniform vec3  uColor;
uniform float uGain;
uniform float uOpacity;

in float vAge;
in float vPulse;
in float vHead;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;
    float core = 1.0 - smoothstep(0.0, 0.6, r);
    float halo = pow(1.0 - r, 2.0);
    vec3 rgb = uColor * (0.7 + 0.9 * vHead + 1.4 * vPulse) * (0.5 + halo) + vec3(1.0) * core * (0.3 + 0.7 * vHead);
    vec4 color = vec4(rgb * uGain, clamp(0.4 + halo, 0.0, 1.0) * uOpacity);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
