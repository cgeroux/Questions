#include <cmath>


//calculate distance between two particles in 2-dimension
double distance_2d(double self_rx, double self_ry, double neigh_rx, double neigh_ry){
    return sqrt((self_rx - neigh_rx) * (self_rx - neigh_rx) + (self_ry - neigh_ry) * (self_ry - neigh_ry)); 
}


//calculate distance between two particles in 2-dimension based on periodic boundaries
/*
xbound: length of the simulation box boundary in x direction
ybound: length of the simulation box boundary in y direction
*/
double distance_2d_periodic(double self_rx, double self_ry, double neigh_rx, double neigh_ry, double xbound, double ybound){
    
    double xdistance = fabs(self_rx - neigh_rx);
    if(xdistance > 0.5 * xbound){xdistance = xbound - xdistance;}

    double ydistance = fabs(self_ry - neigh_ry);
    if(ydistance > 0.5 * ybound){ydistance = ybound - ydistance;}

    return sqrt(xdistance * xdistance + ydistance * ydistance);

}


//f(q) in equation (B2) in https://academic.oup.com/mnras/article/401/3/1475/1089493
double f(double q){
    return (1 - (3.0 / 2.0) * q * q + (3.0 / 4.0) * q * q * q) * (0 <= q) * (q < 1) + (1.0 / 4.0) * pow(2 - q, 3) * (1 <= q) * (q < 2);
}


//f'(q)
//derivative of f(q) in equation (B2) in https://academic.oup.com/mnras/article/401/3/1475/1089493
double f1(double q){
    return (q * (9.0 * q / 4.0 - 3.0)) * (0 <= q) * (q < 1) + (-3.0 * (q - 2) * (q - 2) / 4.0) * (1 <= q) * (q < 2);
}


//Fab in equation (B5) in https://academic.oup.com/mnras/article/401/3/1475/1089493
//h: smoothing length
//distance: distance between two particles a and b
//d: number of spatial dimensions
double fab(double h, double distance, int d){ 
    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3);
    double q = distance / h;
    return norm * f1(q) / pow(h, d + 1);
}


//cubic spline kernel w in equation (6) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
double w(double q, int d){ //q = distance / h, d: number of spatial dimensions

    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3);
    return norm * f(q);
    
}


/*
density: density at this particle
u: internal energy at this particle
*/
//pressure at this particle
//P=(γ−1)ρu in ieos(2) in https://phantomsph.readthedocs.io/en/latest/eos.html
double p(double density, double u){
    
    double gamma = 5.0 / 3.0; // γ in P=(γ−1)ρu in ieos(2) in https://phantomsph.readthedocs.io/en/latest/eos.html   γ = 5/3 here
    return (gamma - 1) * density * u;

}


//the speed of sound in an ideal gas in https://en.wikipedia.org/wiki/Speed_of_sound
double cal_cs(double u){//u: internal energy of the particle
    
    double gamma = 5.0 / 3.0; // γ 
    double cs = sqrt(gamma * (gamma - 1) * u);
    return cs;

}


//calculate position coordinate difference in the same direction between 2 particles, such as xa - xb, ya - yb, za - zb in periodic boundary.
//coor_a: position coordinate of particle a in one direction (x or y or z direction)
//coor_b: position coordinate of particle b in the direction as a (x or y or z direction)
//bound: the length of the boundary of the particle lattice in that direction
double coor_difference(double coor_a, double coor_b, double bound){
    double difference = coor_a - coor_b;
    if(fabs(difference) > 0.5 * bound){
        if(difference > 0){
            difference -= bound;
        }
        else{
            difference += bound;
        }
    }
    return difference;
}


//vab · rab in equation (103) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in 2-dimension with considering periodic boundary
double vabrab(double vax, double vay, double vbx, double vby, double xa, double ya, double xb, double yb, double xbound, double ybound){

    double distance = distance_2d_periodic(xa, ya, xb, yb, xbound, ybound);
    double x_difference = coor_difference(xa, xb, xbound); //xa - xb in periodic boundary
    double y_difference = coor_difference(ya, yb, ybound); //ya - yb in periodic boundary
    return (((vax - vbx) * x_difference + (vay - vby) * y_difference) / distance);

}


//signal velosity vsig in equation (103) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in 2-dimension with considering periodic boundary
// There are 2 particles: praticle a and particle b in this equation
// ua: internal energy u of particle a
// vax: velosity of particle a in x direction
// xa: x coordinate of particle a
double v_sig(double ua, double ub, double vax, double vay, double vbx, double vby, double xa, double ya, double xb, double yb, double xbound, double ybound){
    
    double cs_a = cal_cs(ua);
    double cs_b = cal_cs(ub);
    double vr = vabrab(vax, vay, vbx, vby, xa, ya, xb, yb, xbound, ybound); //vab · rab in equation (103) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
    if(vr <= 0){return 0.5 * (cs_a + cs_b - 2 * vabrab(vax, vay, vbx, vby, xa, ya, xb, yb, xbound, ybound));}
    return 0;

}


/*
self_rx, self_ry: x and y coordinates of the particle itself
neigh_rx, neigh_ry: x and y coordinates of neighboring particle
h: smoothing length
d: number of spatial dimensions
*/
//cubic spline smoothing kernel W in equation (B1) in https://academic.oup.com/mnras/article/401/3/1475/1089493
double cubic_spline_kernel(double self_rx, double self_ry, double neigh_rx, double neigh_ry, double h, double d){ 

    double distance = distance_2d(self_rx, self_ry, neigh_rx, neigh_ry); //distance between two particles
    double q = distance / h; // variable q in the math formulation
    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3); // σ normalisation factor

    return (norm / pow(h, d)) * f(q);

}


/*
self_rx, self_ry: x and y coordinates of the particle itself
neigh_rx, neigh_ry: x and y coordinates of neighboring particle
h: smoothing length
d: number of spatial dimensions
*/
//first derivative of cubic spline smoothing kernel W with respect to h, (B6) in https://academic.oup.com/mnras/article/401/3/1475/1089493
double first_derivative_kernel(double self_rx, double self_ry, double neigh_rx, double neigh_ry, double h, double d){ 
    
    double distance = distance_2d(self_rx, self_ry, neigh_rx, neigh_ry); //distance between two particles
    double q = distance / h; // variable q in the math formulation
    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3); // σ normalisation factor

    return (-norm / pow(h, d + 1)) * (d * f(q) + q * f1(q));

}


/*
self_rx, self_ry: x and y coordinates of the particle itself
neigh_rx, neigh_ry: x and y coordinates of neighboring particle
h: smoothing length
d: number of spatial dimensions
*/
//cubic spline smoothing kernel W in equation (B1) in https://academic.oup.com/mnras/article/401/3/1475/1089493 based on periodic boundary in 2 dimensions
double cubic_spline_kernel_periodic(double self_rx, double self_ry, double neigh_rx, double neigh_ry, double h, double d, double xbound, double ybound){ 

    double distance = distance_2d_periodic(self_rx, self_ry, neigh_rx, neigh_ry, xbound, ybound); //distance between two particles based on periodic boundary
    double q = distance / h; // variable q in the math formulation
    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3); // σ normalisation factor

    return (norm / pow(h, d)) * f(q);

}


/*
self_rx, self_ry: x and y coordinates of the particle itself
neigh_rx, neigh_ry: x and y coordinates of neighboring particle
h: smoothing length
d: number of spatial dimensions
*/
//first derivative of cubic spline smoothing kernel W with respect to h, (B6) in https://academic.oup.com/mnras/article/401/3/1475/1089493 based on periodic boundary in 2 dimensions
double first_derivative_kernel_periodic(double self_rx, double self_ry, double neigh_rx, double neigh_ry, double h, double d, double xbound, double ybound){ 
    
    double distance = distance_2d_periodic(self_rx, self_ry, neigh_rx, neigh_ry, xbound, ybound); //distance between two particles based on periodic boundary
    double q = distance / h; // variable q in the math formulation
    double norm = (2.0 / 3.0) * (d == 1) + (10.0 / (7.0 * M_PI)) * (d == 2) + (1.0 / M_PI) * (d == 3); // σ normalisation factor

    return (-norm / pow(h, d + 1)) * (d * f(q) + q * f1(q));

}