// DEEP OCEAN ROBOT VISION 4D - bathymetry surface (GLSL MAT, vertex). Used for BOTH the real gridded terrain
// (uSource = 0, bilinear height texture) and the sonar-reconstructed seabed (uSource = 1, per-cell texture with
// first-observation time so the map GROWS as the robot surveys).
// Geometry : flat Grid SOP (rows x cols = texture size), uv 0..1, uv.s -> East, uv.t -> North.
// Samplers : sHeight  rgba32float, r = elevation m (true terrain)
//            sRecon   rgba32float, r = elevation m or <-9000 if unobserved, g = first observation time s, b = sigma m, a = count
// Vectors  : uExt(vec4 x0,x1,y0,y1 ENU m), uZRef (m), uExag (visual only), uSource, uTime (s)
uniform sampler2D sHeight;
uniform sampler2D sRecon;
uniform vec4  uExt;
uniform float uZRef;
uniform float uExag;
uniform float uSource;
uniform float uTime;

out vec3  vW;
out vec3  vN;
out float vElev;
out float vValid;
out float vTFirst;
out float vSig;
out float vCount;

vec4 cell(ivec2 px, vec2 st)
{
    if (uSource < 0.5) {
        return vec4(texture(sHeight, st).r, 0.0, 0.0, 1.0);
    }
    vec4 r = texelFetch(sRecon, px, 0);
    return vec4(r.r, r.g, r.b, (r.r > -9000.0 && r.g <= uTime) ? 1.0 : 0.0);
}

void main()
{
    vec2 st = uv[0].st;
    ivec2 sz = textureSize(sRecon, 0);
    if (uSource < 0.5) sz = textureSize(sHeight, 0);
    ivec2 px = clamp(ivec2(floor(st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);

    vec4 c = cell(px, st);
    float h = c.a > 0.5 ? c.x : uZRef;                    // unobserved cells are discarded in the pixel shader

    // central-difference normal in metres
    vec2 stepM = vec2((uExt.y - uExt.x) / float(sz.x - 1), (uExt.w - uExt.z) / float(sz.y - 1));
    vec2 stU = vec2(1.0 / float(sz.x - 1), 0.0);
    vec2 stV = vec2(0.0, 1.0 / float(sz.y - 1));
    ivec2 pxE = clamp(px + ivec2(1, 0), ivec2(0), sz - 1), pxW = clamp(px - ivec2(1, 0), ivec2(0), sz - 1);
    ivec2 pxN = clamp(px + ivec2(0, 1), ivec2(0), sz - 1), pxS = clamp(px - ivec2(0, 1), ivec2(0), sz - 1);
    vec4 cE = cell(pxE, st + stU), cW = cell(pxW, st - stU), cN = cell(pxN, st + stV), cS = cell(pxS, st - stV);
    float hE = cE.a > 0.5 ? cE.x : h, hW = cW.a > 0.5 ? cW.x : h, hN = cN.a > 0.5 ? cN.x : h, hS = cS.a > 0.5 ? cS.x : h;
    vec3 n = normalize(vec3(-(hE - hW) / (2.0 * stepM.x) / max(uExag, 1e-3) * 1.0, 1.0 / max(uExag, 1e-3) * 0.0 + 1.0,
                            (hN - hS) / (2.0 * stepM.y)));
    // TD axes: x = East, y = Up, z = -North.  dh/dE = (hE-hW)/(2dx), dh/dN = (hN-hS)/(2dy) -> N_td = (-dh/dE*exag, 1, +dh/dN*exag)
    n = normalize(vec3(-(hE - hW) / (2.0 * stepM.x) * uExag, 1.0, (hN - hS) / (2.0 * stepM.y) * uExag));

    vec3 p = vec3(mix(uExt.x, uExt.y, st.x), (h - uZRef) * uExag, -mix(uExt.z, uExt.w, st.y));
    vW = p;
    vN = n;
    vElev = h;
    vValid = c.a;
    vTFirst = c.y;
    vSig = c.z;
    vCount = (uSource < 0.5) ? 1.0 : texelFetch(sRecon, px, 0).a;
    gl_Position = TDWorldToProj(TDDeform(p));
}
