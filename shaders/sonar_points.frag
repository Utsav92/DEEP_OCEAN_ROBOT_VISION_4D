// DEEP OCEAN ROBOT VISION 4D - neon / fluorescent LED sonar points (GLSL MAT, pixel).
// Vectors: uColorMode (0 intensity, 1 depth, 2 age, 3 scan bands), uGain, uOpacity, uCinematic (0 scientific, 1 cinematic),
//          uClock (wall-clock s, drives hue cycling ONLY in cinematic), uFreshS (s highlight of a new scan), uDecay (s)
// Scientific mode: colours are a pure function of the measured attributes (no hue cycling, no echo flash).
uniform float uColorMode;
uniform float uGain;
uniform float uOpacity;
uniform float uCinematic;
uniform float uClock;
uniform float uFreshS;
uniform float uDecay;

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

// fully saturated fluorescent ramp: hot magenta -> UV violet -> electric blue -> cyan -> acid green -> yellow
vec3 neon(float x)
{
    return hsv2rgb(vec3(fract(0.93 - 0.76 * clamp(x, 0.0, 1.0)), 1.0, 1.0));
}

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;

    float hueShift = uCinematic * 0.06 * sin(uClock * 0.35);
    vec3 col;
    if (uColorMode < 0.5) {
        col = neon(0.15 + 0.85 * vInt + hueShift);
    } else if (uColorMode < 1.5) {
        col = neon(1.0 - vDepth + hueShift);            // shallow = magenta/pink, deep = cyan/green
    } else if (uColorMode < 2.5) {
        col = neon(clamp(vAge / max(uDecay, 1.0), 0.0, 1.0));   // new = hot magenta, old = yellow
    } else {
        col = neon(fract(vScan * 0.043 + hueShift));    // scan bands cycle the whole neon wheel
    }

    float core = 1.0 - smoothstep(0.0, 0.5, r);
    float halo = pow(1.0 - r, 2.0);
    float fresh = exp(-max(vAge, 0.0) / max(uFreshS, 0.5));
    vec3 rgb = col * (0.55 + 0.9 * vInt) * (0.45 + 0.9 * halo) + vec3(1.0) * core * 0.55;
    rgb += uCinematic * fresh * (vec3(1.0, 0.9, 1.0) * 1.2 + col);   // white-hot flash of the newest scan
    rgb *= uGain;

    float a = clamp(0.35 + 0.65 * halo + core, 0.0, 1.0) * uOpacity * vFade;
    vec4 color = vec4(rgb, a);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
