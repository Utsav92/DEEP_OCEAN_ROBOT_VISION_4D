// DEEP OCEAN ROBOT VISION 4D - neon bloom (GLSL TOP, single pass, soft-limited).
// Input 0: image. Bright pixels (above uThreshold, saturated neon) are blurred on two rings and added back.
// Vectors: uGlow (0 = pass-through), uRadius (pixels), uThreshold (0..1)
uniform float uGlow;
uniform float uRadius;
uniform float uThreshold;

out vec4 fragColor;

vec3 bright(vec2 uv)
{
    vec3 c = texture(sTD2DInputs[0], uv).rgb;
    float l = max(c.r, max(c.g, c.b));
    return c * smoothstep(uThreshold, uThreshold + 0.35, l);
}

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(sTD2DInputs[0], 0));
    vec2 uv = vUV.st;
    vec4 base = texture(sTD2DInputs[0], uv);
    vec3 rgb = base.rgb;
    if (uGlow > 0.0) {
        vec3 acc = bright(uv) * 0.18;
        float wsum = 0.18;
        for (int i = 0; i < 12; i++) {
            float a = 6.2831853 * float(i) / 12.0;
            vec2 d = vec2(cos(a), sin(a)) * texel;
            acc += bright(uv + d * uRadius) * 0.10;
            acc += bright(uv + d * uRadius * 2.4) * 0.05;
            wsum += 0.15;
        }
        vec3 g = acc / wsum * uGlow * 2.2;
        rgb += g / (1.0 + 0.35 * g);
    }
    fragColor = TDOutputSwizzle(vec4(rgb, base.a));
}
