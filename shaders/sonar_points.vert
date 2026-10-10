// DEEP OCEAN ROBOT VISION 4D - calibrated sonar returns as world-space LED points (GLSL MAT, vertex).
// Shared by sonar_points.frag (intensity/depth/age/scan colouring) and uncertainty.frag (sigma colouring).
// Geometry : POINT Grid SOP, 512 x 512 = 262,144 slots, uv 0..1 (rows/cols = texture size).
// Samplers : sPosDR   position TOP, dead-reckoned world TD axes (x=E, y=U-zref, z=-N), a = valid
//            sPosSL   position TOP, SLAM-corrected (same slot order); a = slant range m (0 = unused)
//            sAttr    attribute TOP: r = intensity 0..1, g = observation time s (since first ping), b = sigma_h m, a = ping index
// Vectors  : uTime (observation time s), uMix (0 DR .. 1 SLAM), uPointSize, uRefDist, uMaxRange, uCamPos(vec3),
//            uDensity, uHistoryS (0 = persistent map, >0 = only points younger than this), uExag (visual vertical exaggeration),
//            uExplodeMps (ARTISTIC displacement m per s of age; 0 = measured positions), uDepthLo, uDepthHi (TD y range for depth colour)
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
uniform float uExplodeMps;
uniform float uSonarRange;     // m, hide soundings with a longer slant range (ps.a = slant range)
uniform float uDepthLo;
uniform float uDepthHi;

out float vAge;
out float vInt;
out float vDepth;
out float vSig;
out float vScan;
out float vFade;

void main()
{
    ivec2 sz = textureSize(sPosSL, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);

    vec4 pd = texelFetch(sPosDR, px, 0);
    vec4 ps = texelFetch(sPosSL, px, 0);
    vec4 at = texelFetch(sAttr, px, 0);

    vec3 p = mix(pd.xyz, ps.xyz, uMix);
    float depth01 = clamp((p.y - uDepthLo) / max(uDepthHi - uDepthLo, 1e-3), 0.0, 1.0);
    float age = uTime - at.g;
    p.y *= uExag;
    p.y += max(age, 0.0) * uExplodeMps;

    float dist = distance(p, uCamPos);
    float h = fract(sin(dot(vec2(px), vec2(12.9898, 78.233))) * 43758.5453);
    bool young = (uHistoryS <= 0.0) || (age <= uHistoryS);
    bool visible = (ps.a > 0.0) && (ps.a <= uSonarRange) && (age >= 0.0) && young && (h <= uDensity) && (dist <= uMaxRange);

    vec4 worldPos = TDDeform(p);
    gl_Position = visible ? TDWorldToProj(worldPos) : vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 1.0), 1.5, 6.0) : 0.0;

    vAge = age;
    vInt = at.r;
    vDepth = depth01;
    vSig = at.b;
    vScan = at.a;
    vFade = 1.0 - smoothstep(0.85 * uMaxRange, uMaxRange, dist);
}
