// DEEP OCEAN ROBOT VISION 4D - stylised underwater atmosphere (GLSL TOP). NOT a calibrated optical model.
// Input 0: the 3D render. Adds a violet/blue neon haze gradient, floating plankton sparkle, scanlines and a vignette.
// All terms are zero in scientific mode (uCinematic = 0 returns the input unchanged apart from the vignette amount).
// Vectors: uHaze (0..1), uVignette (0..1), uSparkle (0..1), uScan (0..1), uChroma (pixels), uClock (s)
uniform float uHaze;
uniform float uVignette;
uniform float uSparkle;
uniform float uScan;
uniform float uChroma;
uniform float uClock;

out vec4 fragColor;

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main()
{
    vec2 uv = vUV.st;
    vec2 res = vec2(textureSize(sTD2DInputs[0], 0));
    vec2 dir = (uv - 0.5);
    vec2 ca = dir * uChroma / res;
    vec3 rgb;
    rgb.r = texture(sTD2DInputs[0], uv + ca).r;
    rgb.g = texture(sTD2DInputs[0], uv).g;
    rgb.b = texture(sTD2DInputs[0], uv - ca).b;
    float a = texture(sTD2DInputs[0], uv).a;

    // vertical neon haze: ultraviolet at the top, electric blue-teal toward the floor
    vec3 hazeTop = vec3(0.10, 0.02, 0.30);
    vec3 hazeBot = vec3(0.00, 0.10, 0.22);
    rgb += uHaze * mix(hazeBot, hazeTop, smoothstep(0.0, 1.0, uv.y)) * 0.55;

    // drifting plankton sparkle in fluorescent colours
    vec2 g = floor(uv * res / 6.0);
    float h = hash(g + floor(uClock * 3.0));
    float tw = step(0.9965, h) * (0.5 + 0.5 * sin(uClock * 6.0 + h * 40.0));
    vec3 sc = 0.5 + 0.5 * cos(6.2831853 * (h * 7.0 + vec3(0.0, 0.33, 0.67)));
    rgb += uSparkle * tw * sc * 1.2;

    rgb *= 1.0 - uScan * 0.12 * (0.5 + 0.5 * sin(uv.y * res.y * 3.14159));
    float r = length(dir) * 1.41421356;
    rgb *= 1.0 - uVignette * smoothstep(0.5, 1.15, r);
    fragColor = TDOutputSwizzle(vec4(rgb, a));
}
