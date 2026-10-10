// DEEP OCEAN ROBOT VISION 4D - vehicle trajectory as an LED strip (GLSL MAT, vertex).
// Geometry : POINT Grid SOP matching the path texture (160 x 160 slots, uv 0..1).
// Sampler  : sPath  rgba32float: xyz = TD position, w = observation time s (1e9 = unused slot)
// Vectors  : uTime, uExag, uPointSize, uRefDist, uCamPos(vec3), uTrailS (0 = full history, >0 only the last uTrailS s),
//            uClock (wall s) / uPulseSpeed / uPulseCount drive the travelling chase-light (cinematic only)
uniform sampler2D sPath;
uniform float uTime;
uniform float uExag;
uniform float uPointSize;
uniform float uRefDist;
uniform vec3  uCamPos;
uniform float uTrailS;
uniform float uClock;
uniform float uPulseSpeed;
uniform float uPulseCount;
uniform float uCinematic;

out float vAge;
out float vPulse;
out float vHead;

void main()
{
    ivec2 sz = textureSize(sPath, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);
    vec4 q = texelFetch(sPath, px, 0);
    float idx = float(px.y * sz.x + px.x);
    float age = uTime - q.w;
    bool visible = (age >= 0.0) && (uTrailS <= 0.0 || age <= uTrailS) && (q.w < 1e8);

    vec3 p = q.xyz;
    p.y *= uExag;
    float dist = distance(p, uCamPos);
    gl_Position = visible ? TDWorldToProj(TDDeform(p)) : vec4(2.0, 2.0, 2.0, 1.0);
    float head = exp(-max(age, 0.0) / 20.0);                       // the newest stretch is fatter and brighter
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 1.0) * (1.0 + 1.2 * head), 1.5, 8.0) : 0.0;
    vAge = age;
    vHead = head;
    vPulse = uCinematic * (0.5 + 0.5 * sin(6.2831853 * (idx * uPulseCount / float(sz.x * sz.y) - uClock * uPulseSpeed)));
}
