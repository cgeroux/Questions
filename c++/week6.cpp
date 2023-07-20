#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include "tools.h"

using namespace std;


//structure used to store vetcors of attributes of particles
struct Coordinates {
    
    vector<double> rx; //x coordinates
    vector<double> ry; //y coordinates
    vector<double> rz; //z coordinates
    
    vector<double> vx; //x velosity
    vector<double> vy; //y velosity
    vector<double> vz; //z velosity

    vector<double> dvxdt; //equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in x direction
    vector<double> dvydt; //equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in y direction
    vector<double> dvzdt; //equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in z direction
    
    vector<double> u; //internal energy u in equation (2.72) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    vector<double> dudt; //discretised internal energy (time derivative of u) in equation (2.74) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf

    vector<double> h; //smoothing length
    vector<double> m; //mass
    vector<double> density; // density at each particle
    vector<double> p; // pressure at each particle
    vector<double> omega; // Ωa in equation (2.42) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    vector<double> temp_timestep; //store temporary timestep of each particle; we need to find the minimal one from this vector to be the global timestep for every particle

    double timestep; //global timestep for every particle
    int dimensions; // number of spatial dimensions in this program
    double x_min; //the minimal value of x coordinate of the particle lattice
    double y_min; //the minimal value of y coordinate of the particle lattice
    double xbound; // length of simulation box boundary in x direction
    double ybound; // length of simulation box boundary in y direction
    double zbound; // length of simulation box boundary in z direction

} coordinates;


/*
nx: number of particles per row and per column, the number of total particles is nx*nx;
*/
//create a lattice of particles in square arrangement
Coordinates lattice_init_square(int nx = 50){

    //initialization
    coordinates.rx.assign(nx * nx, 0);
    coordinates.ry.assign(nx * nx, 0);
    //coordinates.rz.assign(nx * nx, 0);

    coordinates.vx.assign(nx * nx, 0);
    coordinates.vy.assign(nx * nx, 0);
    //coordinates.vz.assign(nx * nx, 0);

    coordinates.dvxdt.assign(nx * nx, 0);
    coordinates.dvydt.assign(nx * nx, 0);
    //coordinates.dvzdt.assign(nx * nx, 0);

    coordinates.u.assign(nx * nx, 1); //I set u = 1 in this case
    coordinates.dudt.assign(nx * nx, 0);

    coordinates.h.assign(nx * nx, 0.02); // h = 1/50 in this program as discussed
    coordinates.m.assign(nx * nx, 0.43); //I set all praticles' mass to 0.43
    coordinates.density.assign(nx * nx, 0);
    coordinates.p.assign(nx * nx, 0);
    coordinates.omega.assign(nx * nx, 0);
    coordinates.temp_timestep.assign(nx * nx, 0);

    coordinates.timestep = 0;
    coordinates.dimensions = 2; //2 dimensions in this case
    coordinates.x_min = 0; // 0 in this case
    coordinates.y_min = 0; // 0 in this case
    coordinates.xbound = 1;
    coordinates.ybound = 1;
    
    double offset = 0.5 / nx;
    int counter = 0;

    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nx; j++){
            coordinates.rx[counter] = (double)i / nx + offset;
            coordinates.ry[counter] = (double)j / nx + offset;
            // if (coordinates.rx[counter] < 0.5){
            //     coordinates.u[counter] = 1.0;
            // } 
            // else{
            //     coordinates.u[counter] = 2.0;
            // }   
            counter++;
        }
    }

    return coordinates;

}


//calculate and store density to coordinates vector and return it
Coordinates calculate_density(Coordinates coordinates){

    for(int i = 0; i < coordinates.rx.size(); i++){
        double d = 0; //temporary variable d to store density at each particle
        for(int j = 0; j < coordinates.rx.size(); j++){
            d += (coordinates.m[j] * cubic_spline_kernel_periodic(coordinates.rx[i], coordinates.ry[i], coordinates.rx[j], coordinates.ry[j], coordinates.h[i], coordinates.dimensions, coordinates.xbound, coordinates.ybound));
        }
        coordinates.density[i] = d;
    }

    return coordinates;

}


//set each particle's smoothing length and store them to coordinates vector
Coordinates setting_smoothing_length(Coordinates coordinates){
    
    double f1; // f(h) - equation (2.46) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    double f2; // f'(h) - equation (2.48) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    int lattice_size = coordinates.rx.size();
    for(int i = 0; i < lattice_size; i++){
        
        double mass = coordinates.m[i];
        double h0 = coordinates.h[i]; //value of the smoothing length before the first iteration
        double dimensions = coordinates.dimensions;
        double omega;
        double density = coordinates.density[i];

        f1 = mass * pow(1.2 / h0, dimensions) - density; //η = 1.2 in equation (2.46) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
        
        //calculate current Ω
        omega = 1;
        for (int j = 0; j < lattice_size; j++){ //first_derivative_kernel was implemented in tools.h
            omega += ((h0 * coordinates.m[j] * first_derivative_kernel_periodic(coordinates.rx[i], coordinates.ry[i], coordinates.rx[j], coordinates.ry[j], h0, dimensions, coordinates.xbound, coordinates.ybound)) / (density * dimensions)); //euqation(27) and (28) in Price (2012)
        }

        f2 = (-dimensions) * density * omega / h0; // equation (2.49) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf

        double h_old = h0;
        double h_new = h_old - f1 / f2;

        while((fabs(h_new - h_old) / h0) >= 0.0001){
            
            omega = 0;
            density = 0;
            for (int j = 0; j < lattice_size; j++){
                density += (coordinates.m[j] * cubic_spline_kernel_periodic(coordinates.rx[i], coordinates.ry[i], coordinates.rx[j], coordinates.ry[j], h_new, dimensions, coordinates.xbound, coordinates.ybound)); // calculate density based on new h
                omega += (coordinates.m[j] * first_derivative_kernel_periodic(coordinates.rx[i], coordinates.ry[i], coordinates.rx[j], coordinates.ry[j], h_new, dimensions, coordinates.xbound, coordinates.ybound)); // calculate omega based on new h  
            }
            omega = 1 + (omega * h_new) / (density * dimensions);
            
            coordinates.density[i] = density;
            coordinates.omega[i] = omega;
            
            f1 = mass * pow(1.2 / h_new, dimensions) - density;
            f2 = (-dimensions) * density * omega / h_new;
            
            h_old = h_new;
            h_new = h_old - f1 / f2;

        }

        coordinates.h[i] = h_old; 
        
    }

    return coordinates;

}


//calculate accelerated speed dvdt and dudt at each particle, store them to coordinates vector and return it: equation (44) and (45) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
Coordinates calculate_dvdt_dudt(Coordinates coordinates){
    

    double norm = 10.0 / (7.0 * M_PI); // σ normalisation factor = 10/7π in 2 dimensions here
    int lattice_size = coordinates.rx.size();
    int d = coordinates.dimensions;
    double x_bound = coordinates.xbound;
    double y_bound = coordinates.ybound;

    for(int i = 0; i < lattice_size; i++){

        double xa = coordinates.rx[i]; //x coordinate of particle i
        double ya = coordinates.ry[i]; //y coordinate of particle i
        //double za = coordinates.rz[i]; //z coordinate of particle i

        double vax = coordinates.vx[i]; // velosity of particle i in x direction
        double vay = coordinates.vy[i]; // velosity of particle i in y direction
        //double vaz = coordinates.vz[i]; // velosity of particle i in z direction

        double da = coordinates.density[i]; //density of particle i

        double oa = coordinates.omega[i]; //Ω omega of particle i

        double ha = coordinates.h[i]; //smoothing length of particle i

        double ua = coordinates.u[i]; // internal energy of particle i
        
        double pa = p(da, ua); // pressure at particle i

        double ax = 0; //temporary variable ax to store dvdt in x direction at each particle
        double ay = 0; //temporary variable ay to store dvdt in y direction at each particle
        //double az = 0; //temporary variable az to store dvdt in z direction at each particle

        double ut = 0; //temporary variable ut to store dudt at each particle

        for(int j = 0; j < lattice_size; j++){

            if(i != j){

                double xb = coordinates.rx[j]; //x coordinate of particle j
                double yb = coordinates.ry[j]; //y coordinate of particle j
                //double zb = coordinates.rz[j]; //z coordinate of particle j

                double vbx = coordinates.vx[j]; // velosity of particle j in x direction
                double vby = coordinates.vy[j]; // velosity of particle j in y direction
                //double vbz = coordinates.vz[j]; // velosity of particle j in z direction

                double db = coordinates.density[j]; //density of particle j

                double ob = coordinates.omega[j]; //Ω omega of particle j

                double hb = coordinates.h[j]; //smoothing length of particle j

                double ub = coordinates.u[j]; // internal energy of particle j
        
                double pb = p(db, ub); // pressure at particle j

                double mb = coordinates.m[j]; //mass of particle j
            
                double distance = distance_2d_periodic(xa, ya, xb, yb, x_bound, y_bound); //distance betwwen particle a and particle b in periodic boundary

                double vsig = v_sig(ua, ub, vax, vay, vbx, vby, xa, ya, xb, yb, x_bound, y_bound); 

                double vr = vabrab(vax, vay, vbx, vby, xa, ya, xb, yb, x_bound, y_bound);

                double average_fab = (fab(ha, distance, d) + fab(hb, distance, d)) / 2; //(Fab(ha) + Fab(hb)) / 2

                double average_density = (da + db) / 2;       

                
                //calculate dv/dt, du/dt based on equation (44) and (45) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 with considering periodic boundary
                double x_difference = coor_difference(xa, xb, x_bound); // xa - xb in periodic boundary
                double y_difference = coor_difference(ya, yb, y_bound); // ya - yb in periodic boundary


                //calculate dvdt, equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
                ax -= (mb * x_difference * norm * (pa * f1(distance / ha) / (oa * da * da * pow(ha, d + 1)) + pb * f1(distance / hb) / (ob * db * db * pow(hb, d + 1))) / distance);
                ay -= (mb * y_difference * norm * (pa * f1(distance / ha) / (oa * da * da * pow(ha, d + 1)) + pb * f1(distance / hb) / (ob * db * db * pow(hb, d + 1))) / distance);
            
                //calculate dudt, equation (45) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
                ut += (pa * mb * norm * f1(distance / ha) * ((vax - vbx) * x_difference + (vay - vby) * y_difference) / (distance * pow(ha, d + 1) * oa * da * da));
                

                
                //implement artificial viscosity
                //update dvdt, equation (101) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
                ax += (mb * vsig * vr * average_fab * x_difference / (distance * average_density)); //dv/dt in x direction
                ay += (mb * vsig * vr * average_fab * y_difference / (distance * average_density)); //dv/dt in y direction
            
                //update dudt, equation (104) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
                //Read this please: https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis-errata.pdf
                ut -= (mb * average_fab * (0.5 * vsig * vr * vr - sqrt(fabs(pa - pb) / average_density) * (ua - ub)) / average_density);
                //ut += (mb * average_fab * (0.5 * vsig * vr * vr + sqrt(fabs(pa - pb) / average_density) * (ua - ub)) / average_density);
            }
            
        }
        
        coordinates.p[i] = pa;

        coordinates.dvxdt[i] = ax;
        coordinates.dvydt[i] = ay;

        coordinates.dudt[i] = ut;

        //cout << "particle " << i << ": (" << coordinates.rx[i] << ", " << coordinates.ry[i] << "), ax: " << coordinates.dvxdt[i] << ", ay: " << coordinates.dvydt[i] << ", dudt: " << coordinates.dudt[i] << endl;

    }

    return coordinates;
}


//implement leapfrog integration
Coordinates leapfrog_integrator(Coordinates coordinates){
    
    int lattice_size = coordinates.rx.size();
    double xmin = coordinates.x_min;
    double ymin = coordinates.y_min;
    double x_bound = coordinates.xbound;
    double y_bound = coordinates.ybound;
    
    double dt; // global timestep ∆t
    double cs; // speed of sound for each particle


    for(int i = 0; i < lattice_size; i++){ //calculare ∆t for every particle and store temporary ∆t in vector temp_timestep
        cs = cal_cs(coordinates.u[i]);
        coordinates.temp_timestep[i] = 0.25 * coordinates.h[i] / cs; // equation (2.116) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    }
    coordinates.timestep = *min_element(coordinates.temp_timestep.begin(), coordinates.temp_timestep.end());// very first initial timestep ∆t
    dt = coordinates.timestep; // temporary variable to store the global timestep ∆t


    for(int i = 0; i < 10; i++){
        
        for(int j = 0; j < lattice_size; j++){ //update internal energy, velosity and position coordinates of every particle
            
            coordinates.u[j] += (coordinates.dudt[j] * dt / 2); // u(i + 1/2) = u(i) + (du/dt)(i) * ∆t / 2
            coordinates.vx[j] += (coordinates.dvxdt[j] * dt / 2); // v(i + 1/2) = v(i) + (dv/dt)(i) * ∆t / 2  in x direction
            coordinates.vy[j] += (coordinates.dvydt[j] * dt / 2); // v(i + 1/2) = v(i) + (dv/dt)(i) * ∆t / 2  in y direction
            
            //update positions without considering periodic boundary
            coordinates.rx[j] += (coordinates.vx[j] * dt); // x(i + 1) = x(i) + v(i + 1/2) * ∆t, update positions in x direction
            coordinates.ry[j] += (coordinates.vy[j] * dt); // y(i + 1) = y(i) + v(i + 1/2) * ∆t, update positions in y direction

            //update x position with considering periodic boundary
            while(coordinates.rx[j] < xmin){
                coordinates.rx[j] += x_bound;
            }
            while(coordinates.rx[j] > xmin + x_bound){
                coordinates.rx[j] -= x_bound;
            }

            //update y position with considering periodic boundary
            while(coordinates.ry[j] < ymin){
                coordinates.ry[j] += y_bound;
            }
            while(coordinates.ry[j] > ymin + y_bound){
                coordinates.ry[j] -= y_bound;
            }
            
        }

        coordinates = calculate_density(coordinates); //recalculate density
        coordinates = setting_smoothing_length(coordinates); //recalculate density, smoothing length and omega
        coordinates = calculate_dvdt_dudt(coordinates); //recalculate dv/dt, du/dt and pressure

        // recalculate ∆t based on new positions generated above
        for(int j = 0; j < lattice_size; j++){ //calculare ∆t for every particle and store temporary ∆t in vector temp_timestep
            cs = cal_cs(coordinates.u[j]);
            //if(coordinates.h[j] <= 0){cout << "h <= 0! cs = " << cs << ", " << "h = " << coordinates.h[j] << ", u = " << coordinates.u[j] << endl;} //for debugging
            //if(coordinates.u[j] <= 0){cout << "u <= 0! cs = " << cs << ", " << "h = " << coordinates.h[j] << ", u = " << coordinates.u[j] << endl;} //for debugging
            coordinates.temp_timestep[j] = 0.25 * coordinates.h[j] / cs; // equation (2.116) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
        }
        coordinates.timestep = *min_element(coordinates.temp_timestep.begin(), coordinates.temp_timestep.end());// very first initial timestep ∆t
        dt = coordinates.timestep; // temporary variable to store the global timestep ∆t
        
        cout << "dt: " << dt << endl; // for debugging
        
        for(int j = 0; j < lattice_size; j++){ //update internal energy u and velosity v for every particle
            
            coordinates.u[j] += (coordinates.dudt[j] * dt / 2); // u(i + 1) = u(i + 1/2) + (du/dt)(i + 1) * ∆t / 2
            coordinates.vx[j] += (coordinates.dvxdt[j] * dt / 2); // v(i + 1) = v(i + 1/2) + (dv/dt)(i + 1) * ∆t / 2  in x direction
            coordinates.vy[j] += (coordinates.dvydt[j] * dt / 2); // v(i + 1) = v(i + 1/2) + (dv/dt)(i + 1) * ∆t / 2  in y direction
            
        }

    }

    return coordinates;

}


//write particle data to csv
void write_file(){
    
    ofstream myfile;
    myfile.open ("particles.csv");
    myfile << "particle #,x,y,vx,vy,m,Omega,h,density,pressure,dvx/dt,dvy/dt,du/dt\n"; //h: smoothing length
    
    for(int i = 0; i < coordinates.rx.size(); i++){
        myfile << "particle " << i << "," << coordinates.rx[i] << "," << coordinates.ry[i] << "," << coordinates.vx[i] << "," << coordinates.vy[i] << "," << coordinates.m[i] << "," << coordinates.omega[i] << "," << coordinates.h[i] << "," << coordinates.density[i] << "," << coordinates.p[i] << "," << coordinates.dvxdt[i] << "," << coordinates.dvydt[i] << "," << coordinates.dudt[i] << "\n";
    }

    myfile.close();
}


int main(){

    coordinates = lattice_init_square(); //create a lattice of particles in square arrangement
    coordinates = calculate_density(coordinates); // calculate praticles' density and store them
    coordinates = setting_smoothing_length(coordinates); //setting particles' smoothing lengths and print out
    coordinates = calculate_dvdt_dudt(coordinates); //calculate accelerated speed at each particle and store them
    coordinates = leapfrog_integrator(coordinates); //leapfrog integration
    write_file();
    return 0;
    
}