#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>


// constants
#define SIGMA 10.0
#define EPSILON 2.0
#define DT 0.004

#define N1 10
#define N2 20
#define PARTICLES (N1 * N2)



#define RCUT 3.0


#define RLX 20.0
#define RLY 40.0
#define VMAX 1.0
#define INIT_SPACING 1.12
#define RW 4.96
#define RH 10.0

#define ENABLE_GRAVITY true
#define GRAVITY 0.1
#define ENABLE_TEMP_SCALING true
#define TEMP_SCALE 1.5 // 1 by default. 1 is also the state at which it doesnt take energy nor give any away. 


#define CIRCLE_SEGMENTS 20  // Higher = smoother circles
#define PARTICLE_RADIUS 0.025f


double x[PARTICLES], y[PARTICLES];
double vx[PARTICLES], vy[PARTICLES];
double ax[PARTICLES], ay[PARTICLES];
double v_mod[PARTICLES];
double vcorr[PARTICLES];

void initialize_particles() {
    int idx = 0;
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            x[idx] = i * INIT_SPACING + RW;
            y[idx] = j * INIT_SPACING + RH;
            vx[idx] = 0; // initializing statically
            vy[idx] = 0;
            //vx[idx] = (2.0 * rand() / RAND_MAX - 1.0) * VMAX;
            //vy[idx] = (2.0 * rand() / RAND_MAX - 1.0) * VMAX;
            idx++;
        }
    }
}

void compute_forces() {
    double ax_private[PARTICLES] = {0.0};
    double ay_private[PARTICLES] = {0.0};

    #pragma omp parallel
    {
        double ax_local[PARTICLES] = {0.0};
        double ay_local[PARTICLES] = {0.0};

        #pragma omp for schedule(dynamic)
        for (int k = 0; k < PARTICLES - 1; k++) {
            for (int i = k + 1; i < PARTICLES; i++) {
                double dx = x[k] - x[i];
                double dy = y[k] - y[i];
                double dist2 = dx * dx + dy * dy;

                if (dist2 < RCUT) {
                    double rd = (2.0 / pow(dist2, 6) - 1.0) / pow(dist2, 8);
                    double fx = rd * dx;
                    double fy = rd * dy;

                    ax_local[k] += fx;
                    ay_local[k] += fy;
                    ax_local[i] -= fx;
                    ay_local[i] -= fy;
                }
            }
        }

        // Reduce results from local to global
        #pragma omp critical
        {
            for (int i = 0; i < PARTICLES; i++) {
                ax_private[i] += ax_local[i];
                ay_private[i] += ay_local[i];
            }
        }
    }

    // Copy to main arrays
    for (int i = 0; i < PARTICLES; i++) {
        ax[i] = ax_private[i];
        ay[i] = ay_private[i];
        if (ENABLE_GRAVITY) {
            ay[i] -= GRAVITY;
        }
    }
}



void velocity_verlet_step() {
    #pragma omp parallel for
    for (int i = 0; i < PARTICLES; i++) {
        double new_x = x[i] + vx[i] * DT + 0.5 * ax[i] * DT * DT;
        double new_y = y[i] + vy[i] * DT + 0.5 * ay[i] * DT * DT;
        double half_vx = vx[i] + 0.5 * ax[i] * DT;
        double half_vy = vy[i] + 0.5 * ay[i] * DT;
        vcorr[i] = 0;

        if (new_x < 0.0) {
            new_x = -new_x;
            vx[i] = -half_vx;
            vcorr[i] = 1;
        } else if (new_x >= RLX) {
            new_x = 2 * RLX - new_x;
            vx[i] = -half_vx;
            vcorr[i] = 1;
        } else {
            vx[i] = half_vx;
        }

        if (new_y < 0.0) {
            new_y = -new_y;
            vy[i] = -half_vy;
            vcorr[i] = 1;
        } else if (new_y >= RLY) {
            new_y = 2 * RLY - new_y;
            vy[i] = -half_vy;
            vcorr[i] = 1;
        } else {
            vy[i] = half_vy;
        }

        x[i] = new_x;
        y[i] = new_y;
    }

    compute_forces();

    #pragma omp parallel for
    for (int i = 0; i < PARTICLES; i++) {
        vx[i] += 0.5 * ax[i] * DT;
        vy[i] += 0.5 * ay[i] * DT;

        v_mod[i] = sqrt(vx[i] * vx[i] + vy[i] * vy[i]);

        if (ENABLE_TEMP_SCALING && vcorr[i] == 1 && v_mod[i] != 0) {
            vx[i] = TEMP_SCALE * vx[i] / v_mod[i];
            vy[i] = TEMP_SCALE * vy[i] / v_mod[i];
        }
    }
}



int main() {
    initialize_particles();
    compute_forces();

    int steps = 5000;

    double start_time = omp_get_wtime();  // Start timing

    for (int step = 0; step < steps; step++) {
        velocity_verlet_step();
    }

    double end_time = omp_get_wtime();  // End timing

    printf("SMP Simulation of %d steps took %.6f seconds.\n", steps, end_time - start_time);
    return 0;
}
