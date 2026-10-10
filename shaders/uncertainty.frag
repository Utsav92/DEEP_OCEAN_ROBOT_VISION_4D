// DEEP OCEAN ROBOT VISION 4D - sounding uncertainty colouring (GLSL MAT, pixel; pairs with sonar_points.vert).
// Colours each sounding by its COMPUTED horizontal 1-sigma vSig (metres: sensor model + dead-reckoning growth):
//   fluorescent lime = precise, through yellow/orange, to hot magenta = poor. Vectors: uSigLo, uSigHi (m), uGain, uOpacity
uniform float uSigLo;
uniform float uSigHi;
uniform float uGain;
uniform float uOpacity;

in float vAge;
in float vInt;
in float vDepth;
in float vSig;
in float vScan;
in float vFade;

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
    float u = clamp((vSig - uSigLo) / max(uSigHi - uSigLo, 1e-3), 0.0, 1.0);
    vec3 col = hsv2rgb(vec3(mix(0.26, 0.90, u * u), 1.0, 1.0));   // lime (0.26) -> yellow -> orange -> magenta (0.90)
    float halo = pow(1.0 - r, 2.0);
    vec3 rgb = col * (0.5 + 1.0 * halo) * uGain + vec3(1.0) * (1.0 - smoothstep(0.0, 0.4, r)) * 0.4;
    vec4 color = vec4(rgb, clamp(0.3 + 0.7 * halo, 0.0, 1.0) * uOpacity * vFade);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
