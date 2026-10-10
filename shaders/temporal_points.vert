// DEEP OCEAN ROBOT VISION 4D - temporal scan layers / sonar echo (GLSL MAT, vertex).
// Re-draws the SAME world-space soundings as sonar_points.vert but only those inside the history window
// (uHistoryS = n_scans * ping period) with an age-driven opacity, and optionally displaces them along +y proportional
// to age (uExplodeMps, ARTISTIC "time explosion"; measured positions are untouched when it is 0).
// Samplers/Vectors: as sonar_points.vert, plus uDecay (s) for the exponential fade.
uniform sampler2D sPosDR;
uniform sampler2D sPosSL;
uniform sampler2D sAttr;
uniform float uTime;
uniform float uMix;
uniform float uPointSize;
uniform float uRefDist;
uniform float uMaxRange;
uniform vec3  uCamPos;
uniform float uDensity;
uniform float uHistoryS;
uniform float uExag;
uniform float uSonarRange;
uniform float uExplodeMps;
uniform float uDecay;
uniform float uLayerSeconds;     // scans are grouped into translucent layers of this many seconds

out float vAge;
out float vAlpha;
out float vLayer;
out float vInt;

void main()
{
    ivec2 sz = textureSize(sPosSL, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);
    vec4 pd = texelFetch(sPosDR, px, 0);
    vec4 ps = texelFetch(sPosSL, px, 0);
    vec4 at = texelFetch(sAttr, px, 0);

    vec3 p = mix(pd.xyz, ps.xyz, uMix);
    float age = uTime - at.g;
    p.y *= uExag;
    p.y += max(age, 0.0) * uExplodeMps;

    float dist = distance(p, uCamPos);
    float h = fract(sin(dot(vec2(px), vec2(12.9898, 78.233))) * 43758.5453);
    bool visible = (ps.a > 0.0) && (ps.a <= uSonarRange) && (age >= 0.0) && (age <= uHistoryS) && (h <= uDensity) && (dist <= uMaxRange);

    gl_Position = visible ? TDWorldToProj(TDDeform(p)) : vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 1.0), 1.5, 8.0) : 0.0;
    vAge = age;
    vAlpha = exp(-max(age, 0.0) / max(uDecay, 1.0)) * (1.0 - smoothstep(0.85 * uMaxRange, uMaxRange, dist));
    vLayer = floor(age / max(uLayerSeconds, 1.0));
    vInt = at.r;
}
