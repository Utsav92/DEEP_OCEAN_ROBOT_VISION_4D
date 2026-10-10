// DEEP OCEAN ROBOT VISION 4D - temporal layers: UV-violet -> magenta echoes (GLSL MAT, pixel). Additive blending.
// Vectors: uGain, uOpacity, uCinematic, uClock
uniform float uGain;
uniform float uOpacity;
uniform float uCinematic;
uniform float uClock;

in float vAge;
in float vAlpha;
in float vLayer;
in float vInt;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

vec3 hsv2rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;
    // layers alternate UV violet / hot magenta / electric blue / cyan so separated layers read as distinct ghosts
    float hue = fract(0.78 + 0.075 * vLayer + uCinematic * 0.04 * sin(uClock * 0.4));
    vec3 col = hsv2rgb(vec3(hue, 0.95, 1.0));
    float halo = pow(1.0 - r, 1.7);
    vec3 rgb = col * (0.5 + 0.8 * vInt) * (0.4 + halo) * uGain;
    vec4 color = vec4(rgb, halo * vAlpha * uOpacity);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
