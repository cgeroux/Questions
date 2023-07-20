#include <iostream>
#include <cmath>
#include <fstream>
#include "tools.h"

using namespace std;

#define LATTICE_WIDTH 50 //number of particles per row in the latticle initially
#define LATTICE_HEIGHT 50 //number of particles per column in the latticle initially
#define LATTICE_SIZE 2500 //total number of particles in the particle lattice


struct Coordinates {

    int number;//number of particles in the lattice
    int dimensions;//spatial dimensions
    double timestep; //global timestep for every particle
    double x_min; //the minimal value of x coordinate of the particle lattice
    double y_min; //the minimal value of y coordinate of the particle lattice
    double xbound; // length of simulation box boundary in x direction
    double ybound; // length of simulation box boundary in y direction
    
    double rx[LATTICE_SIZE];//an array of x coordinate of particles
    double ry[LATTICE_SIZE];//an array of y coordinate of particles
    
    double vx[LATTICE_SIZE];//an array of velosity in x direction of particles
    double vy[LATTICE_SIZE];

    double dvxdt[LATTICE_SIZE];//equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in x direction
    double dvydt[LATTICE_SIZE];//equation (44) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 in y direction

    double u[LATTICE_SIZE]; //internal energy u in equation (2.72) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    double dudt[LATTICE_SIZE]; //discretised internal energy (time derivative of u) in equation (2.74) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf

    double h[LATTICE_SIZE];//an array of smoothing lengths of particles
    double m[LATTICE_SIZE];//an array of mass of particles
    double density[LATTICE_SIZE];//an array of density of particles
    double p[LATTICE_SIZE];//pressure at each particle
    double omega[LATTICE_SIZE];//an array of omega values of particles, Ωa in equation (2.42) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    double temp_timestep[LATTICE_SIZE];//store temporary timestep of each particle; we need to find the minimal one from this array to be the global timestep for every particle

} coordinates;


//create a lattice of particles in square arrangement
Coordinates lattice_init_square(int nx = LATTICE_WIDTH, int ny = LATTICE_HEIGHT){

    //initialization
    coordinates.number = LATTICE_SIZE;
    coordinates.dimensions = 2;
    coordinates.timestep = 0;
    coordinates.x_min = 0; // 0 in this case
    coordinates.y_min = 0; // 0 in this case
    coordinates.xbound = 1.0;
    coordinates.ybound = 1.0;


    for(int i = 0; i < LATTICE_SIZE; i++){
        coordinates.u[i] = 1.0;
        coordinates.h[i] = 0.02; 
        coordinates.m[i] = 0.43; 
    }
    
    double x_offset = 0.5 / LATTICE_WIDTH;
    double y_offset = 0.5 / LATTICE_HEIGHT;
    int counter = 0;

    for(int i = 0; i < LATTICE_WIDTH; i++){//create particle lattice
        for(int j = 0; j < LATTICE_HEIGHT; j++){
            coordinates.rx[counter] = (double)i / LATTICE_WIDTH + x_offset;
            coordinates.ry[counter] = (double)j / LATTICE_HEIGHT + y_offset;
            counter++;
        }
    }

    return coordinates;

}


__global__ void calculate_density(Coordinates* d_out, Coordinates* d_in){// based on equation(2) in https://www.sciencedirect.com/science/article/abs/pii/S0021999110006753

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int lattice_size = d_in->number;
    
    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->rx[thread_id] = d_in->rx[thread_id];
    d_out->ry[thread_id] = d_in->ry[thread_id];

    d_out->vx[thread_id] = d_in->vx[thread_id];
    d_out->vy[thread_id] = d_in->vy[thread_id];

    d_out->dvxdt[thread_id] = d_in->dvxdt[thread_id];
    d_out->dvydt[thread_id] = d_in->dvydt[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id];
    d_out->dudt[thread_id] = d_in->dudt[thread_id];

    d_out->h[thread_id] = d_in->h[thread_id];
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->p[thread_id] = d_in->p[thread_id];
    d_out->omega[thread_id] = d_in->omega[thread_id];
    d_out->temp_timestep[thread_id] = d_in->temp_timestep[thread_id];

    double d = d_in->density[thread_id];
    for(int i = 0; i < lattice_size; i++){
        d += (d_in->m[i] * cubic_spline_kernel_periodic(d_in->rx[thread_id], d_in->ry[thread_id], d_in->rx[i], d_in->ry[i], d_in->h[thread_id], d_in->dimensions, d_in->xbound, d_in->ybound));
    }
    
    d_out->density[thread_id] = d;

}


//set each particle's smoothing length and store them to coordinates vector
__global__ void setting_smoothing_length(Coordinates* d_out, Coordinates* d_in){//based on 2.2.2 in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int lattice_size = d_in->number;

    double f1; // f(h) - equation (2.46) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    double f2; // f'(h) - equation (2.48) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf
    
    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->rx[thread_id] = d_in->rx[thread_id];
    d_out->ry[thread_id] = d_in->ry[thread_id];

    d_out->vx[thread_id] = d_in->vx[thread_id];
    d_out->vy[thread_id] = d_in->vy[thread_id];

    d_out->dvxdt[thread_id] = d_in->dvxdt[thread_id];
    d_out->dvydt[thread_id] = d_in->dvydt[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id];
    d_out->dudt[thread_id] = d_in->dudt[thread_id];
    
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->p[thread_id] = d_in->p[thread_id];
    d_out->temp_timestep[thread_id] = d_in->temp_timestep[thread_id];

    double mass = d_in->m[thread_id];
    double h0 = d_in->h[thread_id];//value of the smoothing length before the first iteration
    double dimensions = d_in->dimensions;
    double omega;
    double density = d_in->density[thread_id];

    f1 = mass * pow(1.2 / h0, dimensions) - density; //η = 1.2 in equation (2.46) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf

    //calculate current Ω
    omega = 1;
    for (int i = 0; i < lattice_size; i++){ //first_derivative_kernel was implemented in tools.h
        omega += ((h0 * d_in->m[i] * first_derivative_kernel_periodic(d_in->rx[thread_id], d_in->ry[thread_id], d_in->rx[i], d_in->ry[i], h0, dimensions, d_in->xbound, d_in->ybound)) / (density * dimensions)); //euqation(27) and (28) in Price (2012)
    }

    f2 = (-dimensions) * density * omega / h0; // equation (2.49) in https://www.cs.mun.ca/~tstricco/papers/Tricco-phdthesis.pdf

    double h_old = h0;
    double h_new = h_old - f1 / f2;

    while((fabs(h_new - h_old) / h0) >= 0.0001){
            
        omega = 0;
        density = 0;
        for (int i = 0; i < lattice_size; i++){
            density += (d_in->m[i] * cubic_spline_kernel_periodic(d_in->rx[thread_id], d_in->ry[thread_id], d_in->rx[i], d_in->ry[i], h_new, dimensions, d_in->xbound, d_in->ybound)); // calculate density based on new h
            omega += (d_in->m[i] * first_derivative_kernel_periodic(d_in->rx[thread_id], d_in->ry[thread_id], d_in->rx[i], d_in->ry[i], h_new, dimensions, d_in->xbound, d_in->ybound)); // calculate omega based on new h  
        }
        omega = 1 + (omega * h_new) / (density * dimensions);
        
        d_out->density[thread_id] = density;
        d_out->omega[thread_id] = omega;
        
        f1 = mass * pow(1.2 / h_new, dimensions) - density;
        f2 = (-dimensions) * density * omega / h_new;
        
        h_old = h_new;
        h_new = h_old - f1 / f2;

    }

    d_out->h[thread_id] = h_old; 

}


//calculate accelerated speed dvdt and dudt at each particle, store them to coordinates vector and return it: equation (44) and (45) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
__global__ void calculate_dvdt_dudt(Coordinates* d_out, Coordinates* d_in){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    double norm = 10.0 / (7.0 * M_PI); // σ normalisation factor = 10/7π in 2 dimensions here
    int lattice_size = d_in->number;
    int d = d_in->dimensions;
    double x_bound = d_in->xbound;
    double y_bound = d_in->ybound;

    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->rx[thread_id] = d_in->rx[thread_id];
    d_out->ry[thread_id] = d_in->ry[thread_id];

    d_out->vx[thread_id] = d_in->vx[thread_id];
    d_out->vy[thread_id] = d_in->vy[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id];
    
    d_out->h[thread_id] = d_in->h[thread_id];
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->density[thread_id] = d_in->density[thread_id];
    d_out->omega[thread_id] = d_in->omega[thread_id];
    d_out->temp_timestep[thread_id] = d_in->temp_timestep[thread_id];
    

    double xa = d_in->rx[thread_id]; //x coordinate of the particle this thread deals with
    double ya = d_in->ry[thread_id]; //y coordinate of the particle this thread deals with
    
    double vax = d_in->vx[thread_id]; // velosity of particle this thread deals with in x direction
    double vay = d_in->vy[thread_id]; // velosity of particle this thread deals with in y direction

    double da = d_in->density[thread_id]; //denisty of the particle this thread deals with

    double oa = d_in->omega[thread_id]; //Ω omega of the particle this thread deals with

    double ha = d_in->h[thread_id]; //smoothing length of the particle this thread deals with

    double ua = d_in->u[thread_id]; // internal energy of the particle this thread deals with

    double pa = p(da, ua); //d_out->p[thread_id] = pa; // pressure at the particle this thread deals with

    double ax = 0; //temporary variable ax to store dvdt in x direction at the particle this thread deals with
    double ay = 0; //temporary variable ay to store dvdt in y direction at the particle this thread deals with

    double ut = 0; //temporary variable ut to store dudt at the particle this thread deals with

    for(int j = 0; j < lattice_size; j++){

        if(thread_id != j){

            double xb = d_in->rx[j]; //x coordinate of particle j
            double yb = d_in->ry[j]; //y coordinate of particle j

            double vbx = d_in->vx[j]; // velosity of particle j in x direction
            double vby = d_in->vy[j]; // velosity of particle j in y direction

            double db = d_in->density[j]; //density of particle j

            double ob = d_in->omega[j]; //Ω omega of particle j

            double hb = d_in->h[j]; //smoothing length of particle j

            double ub = d_in->u[j]; // internal energy of particle j
    
            double pb = p(db, ub); // pressure at particle j

            double mb = d_in->m[j]; //mass of particle j
        
            double distance = distance_2d_periodic(xa, ya, xb, yb, x_bound, y_bound);

            double vsig = v_sig(ua, ub, vax, vay, vbx, vby, xa, ya, xb, yb, x_bound, y_bound); 

            double vr = vabrab(vax, vay, vbx, vby, xa, ya, xb, yb, x_bound, y_bound);

            double average_fab = (fab(ha, distance, d) + fab(hb, distance, d)) / 2.0; //(Fab(ha) + Fab(hb)) / 2

            double average_density = (da + db) / 2.0; 

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

        }
    
    }

    d_out->p[thread_id] = pa;
    d_out->dvxdt[thread_id] = ax;
    d_out->dvydt[thread_id] = ay;
    d_out->dudt[thread_id] = ut;

}


//update temporary timestep for each particle
__global__ void update_timestep(Coordinates* d_out, Coordinates* d_in){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->rx[thread_id] = d_in->rx[thread_id];
    d_out->ry[thread_id] = d_in->ry[thread_id];

    d_out->vx[thread_id] = d_in->vx[thread_id];
    d_out->vy[thread_id] = d_in->vy[thread_id];

    d_out->dvxdt[thread_id] = d_in->dvxdt[thread_id];
    d_out->dvydt[thread_id] = d_in->dvydt[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id];
    d_out->dudt[thread_id] = d_in->dudt[thread_id];
    
    d_out->h[thread_id] = d_in->h[thread_id];
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->density[thread_id] = d_in->density[thread_id];
    d_out->p[thread_id] = d_in->p[thread_id];
    d_out->omega[thread_id] = d_in->omega[thread_id];
    
    double cs = cal_cs(d_in->u[thread_id]);
    d_out->temp_timestep[thread_id] = 0.25 * d_in->h[thread_id] / cs;

}


//the first half of leapfrog integration
__global__ void leapfrog_firsthalf(Coordinates* d_out, Coordinates* d_in){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    double xmin = d_in->x_min;
    double ymin = d_in->y_min;
    double x_bound = d_in->xbound;
    double y_bound = d_in->ybound;

    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->dvxdt[thread_id] = d_in->dvxdt[thread_id];
    d_out->dvydt[thread_id] = d_in->dvydt[thread_id];

    d_out->dudt[thread_id] = d_in->dudt[thread_id];
    
    d_out->h[thread_id] = d_in->h[thread_id];
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->density[thread_id] = d_in->density[thread_id];
    d_out->p[thread_id] = d_in->p[thread_id];
    d_out->omega[thread_id] = d_in->omega[thread_id];
    d_out->temp_timestep[thread_id] = d_in->temp_timestep[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id] + (d_in->dudt[thread_id] * d_in->timestep / 2.0);
    d_out->vx[thread_id] = d_in->vx[thread_id] + (d_in->dvxdt[thread_id] * d_in->timestep / 2.0);
    d_out->vy[thread_id] = d_in->vy[thread_id] + (d_in->dvydt[thread_id] * d_in->timestep / 2.0);
    d_out->rx[thread_id] = d_in->rx[thread_id] + (d_out->vx[thread_id] * d_in->timestep);
    d_out->ry[thread_id] = d_in->ry[thread_id] + (d_out->vy[thread_id] * d_in->timestep);

    //update x position with considering periodic boundary
    while(d_out->rx[thread_id] < xmin){
        d_out->rx[thread_id] += x_bound;
    }
    while(d_out->rx[thread_id] > xmin + x_bound){
        d_out->rx[thread_id] -= x_bound;
    }

    //update y position with considering periodic boundary
    while(d_out->ry[thread_id] < ymin){
        d_out->ry[thread_id] += y_bound;
    }
    while(d_out->ry[thread_id] > ymin + y_bound){
        d_out->ry[thread_id] -= y_bound;
    }

}


//the second half of leapfrog integration
__global__ void leapfrog_secondhalf(Coordinates* d_out, Coordinates* d_in){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    d_out->number = d_in->number;
    d_out->dimensions = d_in->dimensions;
    d_out->timestep = d_in->timestep;
    d_out->x_min = d_in->x_min;
    d_out->y_min = d_in->y_min;
    d_out->xbound = d_in->xbound;
    d_out->ybound = d_in->ybound;

    d_out->rx[thread_id] = d_in->rx[thread_id];
    d_out->ry[thread_id] = d_in->ry[thread_id];

    d_out->dvxdt[thread_id] = d_in->dvxdt[thread_id];
    d_out->dvydt[thread_id] = d_in->dvydt[thread_id];

    d_out->dudt[thread_id] = d_in->dudt[thread_id];
    
    d_out->h[thread_id] = d_in->h[thread_id];
    d_out->m[thread_id] = d_in->m[thread_id];
    d_out->density[thread_id] = d_in->density[thread_id];
    d_out->p[thread_id] = d_in->p[thread_id];
    d_out->omega[thread_id] = d_in->omega[thread_id];
    d_out->temp_timestep[thread_id] = d_in->temp_timestep[thread_id];

    d_out->u[thread_id] = d_in->u[thread_id] + (d_in->dudt[thread_id] * d_in->timestep / 2.0);
    d_out->vx[thread_id] = d_in->vx[thread_id] + (d_in->dvxdt[thread_id] * d_in->timestep / 2.0);
    d_out->vy[thread_id] = d_in->vy[thread_id] + (d_in->dvydt[thread_id] * d_in->timestep / 2.0);

}



//write particle data to csv
void write_file(){
    
    ofstream myfile;
    myfile.open("particles-cuda.csv");
    myfile << "particle #,x,y,vx,vy,m,Omega,h,density,pressure,dvx/dt,dvy/dt,du/dt\n"; //h: smoothing length
    
    for(int i = 0; i < LATTICE_SIZE; i++){
        myfile << "particle " << i << "," << coordinates.rx[i] << "," << coordinates.ry[i] << "," << coordinates.vx[i] << "," << coordinates.vy[i] << "," << coordinates.m[i] << "," << coordinates.omega[i] << "," << coordinates.h[i] << "," << coordinates.density[i] << "," << coordinates.p[i] << "," << coordinates.dvxdt[i] << "," << coordinates.dvydt[i] << "," << coordinates.dudt[i] << "\n";
    }

    myfile.close();
}


int main(){
    
    coordinates = lattice_init_square(); //create a lattice of particles in 1 * 1 square arrangement
    
    //host memory
    Coordinates* h_in = &coordinates;
    Coordinates* h_out = new Coordinates;

    size_t memory_size = sizeof(coordinates);

    //Declare and allocate device memory
    Coordinates* d_in;
    Coordinates* d_out;
    cudaMalloc((void**)&d_in, memory_size); 
    cudaMalloc((void**)&d_out, memory_size);

    cudaMemcpy(d_in, h_in, memory_size, cudaMemcpyHostToDevice);//data transfer from host to device

    calculate_density<<<4, 625>>>(d_out, d_in);//Kernel function, calculate each particle's density. 4 blocks in total; 625 threads per block.
    cudaDeviceSynchronize();
    
    setting_smoothing_length<<<4, 625>>>(d_out, d_out);//set smoothing length
    cudaDeviceSynchronize();

    calculate_dvdt_dudt<<<4, 625>>>(d_out, d_out);//calculate dv/dt and du/dt
    cudaDeviceSynchronize();

    update_timestep<<<4, 625>>>(d_out, d_out);//update value of timestep for each particle the first time
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, memory_size, cudaMemcpyDeviceToHost); //data transfer from device to host

    cudaFree(d_in); cudaFree(d_out);//free allocated device memory

    coordinates = *h_out; //get the contents of coordinates in host's end


    //implement leapfrog integration, starting here, corresponding to Coordinates leapfrog_integrator(Coordinates coordinates) in week5.cpp
    double min = coordinates.temp_timestep[0]; //temporary variable to store the minimal element in the temp_timestep array
    for(int i = 0; i < LATTICE_SIZE; i++){
        if(coordinates.temp_timestep[i] < min){
            min = coordinates.temp_timestep[i];
        }
    }
    coordinates.timestep = min;


    for(int i = 0; i < 10; i++){

        //host memory
        h_in = &coordinates;
        h_out = new Coordinates;

        memory_size = sizeof(coordinates);

        //allocate device memory
        cudaMalloc((void**)&d_in, memory_size); 
        cudaMalloc((void**)&d_out, memory_size);

        cudaMemcpy(d_in, h_in, memory_size, cudaMemcpyHostToDevice);//data transfer from host to device
        
        leapfrog_firsthalf<<<4, 625>>>(d_out, d_in);
        cudaDeviceSynchronize();

        calculate_density<<<4, 625>>>(d_out, d_out);//Kernel function, calculate each particle's density. 4 blocks in total; 625 threads per block.
        cudaDeviceSynchronize();
        
        setting_smoothing_length<<<4, 625>>>(d_out, d_out);//set smoothing length
        cudaDeviceSynchronize();

        calculate_dvdt_dudt<<<4, 625>>>(d_out, d_out);//calculate dv/dt and du/dt
        cudaDeviceSynchronize();

        update_timestep<<<4, 625>>>(d_out, d_out);//update value of timestep for each particle the first time
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, memory_size, cudaMemcpyDeviceToHost); //data transfer from device to host

        cudaFree(d_in); cudaFree(d_out);//free allocated device memory

        coordinates = *h_out; //get the contents of coordinates in host's end


        min = coordinates.temp_timestep[0]; //temporary variable to store the minimal element in the temp_timestep array
        for(int j = 0; j < LATTICE_SIZE; j++){
            if(coordinates.temp_timestep[j] < min){
                min = coordinates.temp_timestep[j];
            }
        }
        coordinates.timestep = min;

        cout << coordinates.timestep << endl;//debug

        //host memory
        h_in = &coordinates;
        h_out = new Coordinates;

        memory_size = sizeof(coordinates);
        
        //allocate device memory
        cudaMalloc((void**)&d_in, memory_size); 
        cudaMalloc((void**)&d_out, memory_size);

        cudaMemcpy(d_in, h_in, memory_size, cudaMemcpyHostToDevice);//data transfer from host to device

        leapfrog_secondhalf<<<4, 625>>>(d_out, d_in);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, memory_size, cudaMemcpyDeviceToHost); //data transfer from device to host

        cudaFree(d_in); cudaFree(d_out);//free allocated device memory

        coordinates = *h_out; //get the contents of coordinates in host's end

    }


    write_file();//write particles' data to a csv file
    
    return 0;
    
}