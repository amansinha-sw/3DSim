#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

struct Task {
    int taskId;
    int coreId;
    int nextTCore;
    int outgoingTrafficSize;

    unsigned long startTick = 0;   // will be assigned in main()
    unsigned long duration = 0;    // assigned after execution
};

struct Step {
    std::vector<Task> tasks;
};

// Read steps file (taskId, coreId, nextTCore, outgoingTrafficSize)
std::vector<Step> readStepsFile(const std::string &filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) 
        throw std::runtime_error("Cannot open steps file: " + filename);

    int numSteps = 0;
    if (!(infile >> numSteps))
        throw std::runtime_error("Missing number of steps");

    std::vector<Step> steps;
    steps.reserve(numSteps);

    for (int s = 0; s < numSteps; s++) {
        int numTasks = 0;
        if (!(infile >> numTasks)) 
            throw std::runtime_error("Missing number of tasks at step " + std::to_string(s));

        Step step;
        step.tasks.reserve(numTasks);

        for (int t = 0; t < numTasks; t++) {
            Task task;
            if (!(infile >> task.taskId >> task.coreId >> task.nextTCore >> task.outgoingTrafficSize))
                throw std::runtime_error("Invalid task data at step " +
                                         std::to_string(s) + ", task " + std::to_string(t));
            step.tasks.push_back(task);
        }

        steps.push_back(std::move(step));
    }

    return steps;
}

// Centralized CSV logging
void logAllRanksTaskStatisticsCSV(const std::vector<Task> &localTasks,
                                  int rank, int step, int worldSize) {
    std::vector<Task> allTasks;

    if (rank == 0) {
        allTasks = localTasks;
        for (int src = 1; src < worldSize; src++) {
            int numTasks = 0;
            MPI_Recv(&numTasks, 1, MPI_INT, src, 1000 + step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (numTasks > 0) {
                std::vector<Task> buffer(numTasks);
                MPI_Recv(buffer.data(), numTasks * sizeof(Task), MPI_BYTE,
                         src, 2000 + step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                allTasks.insert(allTasks.end(), buffer.begin(), buffer.end());
            }
        }

        std::ofstream ofs;
        std::string filename = "task_log_all_ranks.csv";
        if (step == 0) {
            ofs.open(filename, std::ios::out);
            ofs << "Step,TaskID,CoreID,StartTick,Duration,NextTCore,OutgoingTrafficSize\n";
        } else {
            ofs.open(filename, std::ios::app);
        }

        for (const auto &task : allTasks) {
            ofs << step << ","
                << task.taskId << ","
                << task.coreId << ","
                << task.startTick << ","
                << task.duration << ","
                << task.nextTCore << ","
                << task.outgoingTrafficSize << "\n";
        }
        ofs.close();

    } else {
        int numTasks = localTasks.size();
        MPI_Send(&numTasks, 1, MPI_INT, 0, 1000 + step, MPI_COMM_WORLD);
        if (numTasks > 0) {
            MPI_Send(localTasks.data(), numTasks * sizeof(Task), MPI_BYTE, 0, 2000 + step, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Step> steps;
    if (rank == 0) {
        try {
            steps = readStepsFile(argv[1]);
        } catch (const std::exception &ex) {
            std::cerr << "Error: " << ex.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast number of steps
    int numSteps = (rank == 0 ? (int)steps.size() : 0);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) steps.resize(numSteps);

    // Broadcast tasks of each step
    for (int s = 0; s < numSteps; s++) {
        int numTasks = (rank == 0 ? (int)steps[s].tasks.size() : 0);
        MPI_Bcast(&numTasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) steps[s].tasks.resize(numTasks);

        for (int t = 0; t < numTasks; t++) {
            MPI_Bcast(&steps[s].tasks[t], sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }

    // Execute steps
    for (int s = 0; s < numSteps; s++) {
        // Assign startTick for each task
        for (auto &task : steps[s].tasks) {
            task.startTick = static_cast<unsigned long>(MPI_Wtime() * 1000); // ms
        }

        // Execute tasks for this rank
        std::vector<Task> stepTasks;
        for (auto &task : steps[s].tasks) {
            if (task.coreId == rank) {
                auto t0 = MPI_Wtime();
                // Simulate execution (empty function)
                auto t1 = MPI_Wtime();
                task.duration = static_cast<unsigned long>((t1 - t0) * 1000); // duration in ms
                stepTasks.push_back(task);
            }
        }

        // Centralized logging
        logAllRanksTaskStatisticsCSV(stepTasks, rank, s, size);

        // Outgoing traffic
        for (const auto &task : stepTasks) {
            if (task.outgoingTrafficSize > 0 && task.nextTCore >= 0 && task.nextTCore < size) {
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

