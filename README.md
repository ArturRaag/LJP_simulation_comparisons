## Exploring parallelization as an optimization method via LJP simulation.

The original Lennard-Jones particle simulation was initially developed in JavaScript as an educational project. While suitable for demonstrating the underlying physics, its performance quickly became the limiting factor. Simulating even a few hundred particles resulted in very low frame rates due to the computational cost of evaluating pairwise interactions.

To improve performance, the simulation was completely rewritten in **C**, resulting in a substantial speedup. The native implementation made it possible to simulate over **2000 particles in real time** (approximately 20 FPS on my hardware) without any parallelization, compared to the original JavaScript implementation which struggled to achieve even 1 FPS under similar conditions.

To further optimize the simulation, the force calculations were parallelized using **OpenMP**. Since the Lennard-Jones force computation requires evaluating interactions between every pair of particles, the computational complexity scales approximately as **O(N²)**, making it an excellent candidate for multi-threaded execution.

The OpenMP implementation provided an additional **~3× performance improvement** for sufficiently large particle counts.

An interesting observation was that parallelization actually **reduced performance for small systems**. This is expected, as the overhead associated with thread scheduling, synchronization, and workload distribution can outweigh the computational work being parallelized. Only once the number of particle interactions becomes sufficiently large does parallel execution provide a net performance benefit.

One of the key takeaways from this project was that optimization is highly context-dependent. There is rarely a universally "best" optimization strategy; instead, achieving good performance requires understanding where computational bottlenecks occur and selecting techniques that are appropriate for the workload and hardware architecture.
