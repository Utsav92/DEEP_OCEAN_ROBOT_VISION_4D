// DEEP OCEAN ROBOT VISION 4D - measured beam fan (GLSL MAT, vertex).
// Geometry: Script SOP triangles from the sonar head to consecutive valid returns of the CURRENT ping, in TD axes
// (y already minus z_ref). Point colour attribute Cd.r = range fraction 0 (head) .. 1 (return), Cd.g = beam index 0..1.
// Vectors : uExag (visual vertical exaggeration; also applied to y)
uniform float uExag;

out float vR;
out float vB;

void main()
{
    vec3 p = P;
    p.y *= uExag;
    vR = Cd.r;
    vB = Cd.g;
    gl_Position = TDWorldToProj(TDDeform(p));
}
