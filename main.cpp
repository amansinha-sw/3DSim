#include <iostream>
#include "util.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step-0: Initialize steps data from input configuration file
    std::vector<Step> steps;
    if (rank == 0) {
        steps = readStepsFile(argv[1]);
    }

    // Step-0: Now broadcast steps to all ranks
    broadcastSteps(steps, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Step-0: Set current tick of each rank. Initialize with zero
    unsigned long currentTick = 0;
    int numSteps = steps.size();

    // Step-1: Execute steps
    for (int s = 0; s < numSteps; s++) {
        // Assign startTick for each task
        for (auto &task : steps[s].tasks) {
            task.startTick = currentTick; // ms
        }

        // Execute tasks for this rank
        std::vector<Task> stepTasks;
        for (auto &task : steps[s].tasks) {
            if (task.coreId == rank) {
                auto t0 = MPI_Wtime();
                // Simulate execution (empty function)
                auto t1 = MPI_Wtime();
                task.duration = static_cast<unsigned long>((t1 - t0) * 1000); // duration in ms
                currentTick += task.duration; // Update the current tick for the rank
                stepTasks.push_back(task);
            }
        }

        // Centralized logging
        logAllRanksTaskStatisticsCSV(stepTasks, rank, s, size);

        // Outgoing traffic
        for (const auto &task : stepTasks) {
            if (task.outgoingTrafficSize > 0 &&
                task.nextTCore >= 0 && task.nextTCore < size) {
                std::vector<char> buffer(task.outgoingTrafficSize, 'x');
                MPI_Send(buffer.data(), task.outgoingTrafficSize, MPI_CHAR,
                         task.nextTCore, task.taskId, MPI_COMM_WORLD);
            }
        }
    }

    // Receive any incoming traffic
    MPI_Status status;
    int flag = 0;
    do {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            std::vector<char> buffer(count);
            MPI_Recv(buffer.data(), count, MPI_CHAR, status.MPI_SOURCE,
                     status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } while (flag);

    MPI_Finalize();
    return 0;
}

