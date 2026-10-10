// DEEP OCEAN ROBOT VISION 4D - neon holographic seabed (GLSL MAT, pixel).
// Vectors: uColorMode (0 depth, 1 uncertainty sigma, 2 slope), uContour (m between iso-lines), uGridStep (m LED mesh),
//          uWire (0/1 dense mesh), uDepthLo/uDepthHi (elevation m for the colour ramp), uCamPos(vec3),
//          uCinematic, uClock, uTime, uFreshS, uGain, uOpacity, uSigHi (m), uSource (0 true terrain, 1 reconstruction),
//          uBase (0..1 brightness of the dark base surface), uDuration (s; colour mode 3 = WHEN the cell was first observed)
uniform float uColorMode;
uniform float uContour;
uniform float uGridStep;
uniform float uWire;
uniform float uDepthLo;
uniform float uDepthHi;
uniform vec3  uCamPos;
uniform float uCinematic;
uniform float uClock;
uniform float uTime;
uniform float uFreshS;
uniform float uGain;
uniform float uOpacity;
uniform float uSigHi;
uniform float uSource;
uniform float uBase;
uniform float uDuration;     // s, survey duration for the observation-time colour ramp

in vec3  vW;
in vec3  vN;
in float vElev;
in float vValid;
in float vTFirst;
in float vSig;
in float vCount;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

vec3 hsv2rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 neon(float x)
{
    return hsv2rgb(vec3(fract(0.93 - 0.76 * clamp(x, 0.0, 1.0)), 1.0, 1.0));
}

float lineMask(float v, float period)
{
    float q = v / period;
    float d = abs(fract(q - 0.5) - 0.5) / max(fwidth(q), 1e-5);
    return 1.0 - clamp(d - 0.4, 0.0, 1.0);
}

void main()
{
    TDCheckDiscard();
    if (vValid < 0.999) discard;

    vec3 n = normalize(vN);
    vec3 V = normalize(uCamPos - vW);
    float ndl = clamp(dot(n, normalize(vec3(-0.4, 0.8, 0.45))), 0.0, 1.0);
    float rim = pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 2.5);

    float depth01 = clamp((vElev - uDepthLo) / max(uDepthHi - uDepthLo, 1e-3), 0.0, 1.0);
    float hueShift = uCinematic * 0.08 * sin(uClock * 0.25 + depth01 * 3.0);
    vec3 surf;
    if (uColorMode < 0.5) {
        surf = neon(1.0 - depth01 + hueShift);
    } else if (uColorMode < 1.5) {
        float u = clamp(vSig / max(uSigHi, 1e-3), 0.0, 1.0);
        surf = hsv2rgb(vec3(mix(0.26, 0.90, u * u), 1.0, 1.0));
    } else if (uColorMode > 2.5) {
        surf = neon(clamp(vTFirst / max(uDuration, 1.0), 0.0, 1.0));      // magenta = surveyed first ... yellow = last
    } else {
        float slope = 1.0 - clamp(n.y, 0.0, 1.0);
        surf = neon(clamp(slope * 2.2, 0.0, 1.0));
    }

    // dark, deep-violet body lit from one side; the colour lives in the iso-lines, LED mesh and rim light
    vec3 body = vec3(0.015, 0.006, 0.05) + surf * (0.10 + 0.22 * ndl) * uBase;
    float isoL = lineMask(vElev, uContour);
    float isoMajor = lineMask(vElev, uContour * 4.0);
    vec3 iso = neon(1.0 - depth01 + 0.25 + hueShift) * (1.6 * isoL + 2.4 * isoMajor);

    float gx = lineMask(vW.x, uGridStep), gz = lineMask(vW.z, uGridStep);
    vec3 led = (vec3(0.0, 0.95, 1.0) * gx + vec3(1.0, 0.0, 0.8) * gz) * (0.25 + 0.55 * uWire);
    if (uWire > 0.5) {
        led += (vec3(0.0, 0.9, 1.0) * lineMask(vW.x, uGridStep * 0.2) + vec3(1.0, 0.1, 0.7) * lineMask(vW.z, uGridStep * 0.2)) * 0.35;
    }

    vec3 rgb = body + iso + led + surf * rim * 1.2;

    // newly mapped cells flash while the map grows (reconstruction layer only)
    if (uSource > 0.5) {
        float age = uTime - vTFirst;
        float fresh = exp(-max(age, 0.0) / max(uFreshS, 0.5));
        rgb += fresh * (vec3(1.0, 0.95, 1.0) * 0.9 + surf) * (0.4 + 0.6 * uCinematic);
        rgb += surf * 0.35 * clamp(vCount / 40.0, 0.0, 1.0);      // multi-pass cells glow stronger = observed more often
    }
    vec4 color = vec4(rgb * uGain, uOpacity);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
