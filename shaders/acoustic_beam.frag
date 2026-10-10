// DEEP OCEAN ROBOT VISION 4D - acoustic fan as a neon laser curtain (GLSL MAT, pixel; additive).
// The fan GEOMETRY is measured (head -> returns of the current ping). The travelling wavefront rings are a
// CINEMATIC effect (uCinematic = 1) and do not represent recorded data.
// Vectors: uGain, uOpacity, uCinematic, uClock, uPulse (cycles per second)
uniform float uGain;
uniform float uOpacity;
uniform float uCinematic;
uniform float uClock;
uniform float uPulse;

in float vR;
in float vB;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

vec3 hsv2rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main()
{
    TDCheckDiscard();
    // electric cyan at the head, shifting to hot magenta at the seabed return
    vec3 col = hsv2rgb(vec3(mix(0.50, 0.90, vR), 1.0, 1.0));
    float edge = smoothstep(0.0, 0.08, vB) * smoothstep(1.0, 0.92, vB);
    float ring = uCinematic * pow(0.5 + 0.5 * sin(6.2831853 * (vR * 3.0 - uClock * uPulse)), 6.0);
    float a = (0.10 + 0.25 * vR + 0.9 * ring) * (0.45 + 0.55 * edge);
    vec4 color = vec4(col * (0.6 + 1.6 * ring + 0.8 * vR) * uGain, a * uOpacity);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
