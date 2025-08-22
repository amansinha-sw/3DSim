#include <mpi.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "step.h"
#include <sstream>
#include <limits>

// Read steps file (taskId, coreId, nextTCore, outgoingTrafficSize, appId, appParams)
std::vector<Step> readStepsFile(const std::string &filename) {
    std::ifstream infile(filename);
    if (!infile.is_open())
        throw std::runtime_error("Cannot open steps file: " + filename);

    int numSteps = 0;
    if (!(infile >> numSteps))
        throw std::runtime_error("Missing or invalid number of steps in " + filename);

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
            if (!(infile >> task.taskId >> task.coreId >> task.nextTCore 
                        >> task.outgoingTrafficSize >> task.appId)) {
                throw std::runtime_error("Invalid task data at step " +
                                         std::to_string(s) + ", task " + std::to_string(t));
            }

            task.startTick = 0;
            task.duration = 0;

            // Read rest of line = appParams
            std::string restOfLine;
            std::getline(infile, restOfLine);
            std::istringstream iss(restOfLine);
            int param;
            while (iss >> param) {
                task.appParams.push_back(param);
            }

            step.tasks.push_back(std::move(task));
        }

        steps.push_back(std::move(step));
    }

    return steps;
}

// Broadcast steps data to all ranks
void broadcastSteps(std::vector<Step> &steps, int rank, MPI_Comm comm) {
    int numSteps;
    if (rank == 0) numSteps = steps.size();
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, comm);

    if (rank != 0) steps.resize(numSteps);

    for (int s = 0; s < numSteps; s++) {
        int numTasks;
        if (rank == 0) numTasks = steps[s].tasks.size();
        MPI_Bcast(&numTasks, 1, MPI_INT, 0, comm);

        if (rank != 0) steps[s].tasks.resize(numTasks);

        for (int t = 0; t < numTasks; t++) {
            Task &task = steps[s].tasks[t];

            // Broadcast fixed fields
            MPI_Bcast(&task.taskId, 1, MPI_INT, 0, comm);
            MPI_Bcast(&task.coreId, 1, MPI_INT, 0, comm);
            MPI_Bcast(&task.nextTCore, 1, MPI_INT, 0, comm);
            MPI_Bcast(&task.outgoingTrafficSize, 1, MPI_INT, 0, comm);
            MPI_Bcast(&task.appId, 1, MPI_INT, 0, comm);

            // Broadcast variable-length appParams
            int numParams;
            if (rank == 0) numParams = task.appParams.size();
            MPI_Bcast(&numParams, 1, MPI_INT, 0, comm);

            if (rank != 0) task.appParams.resize(numParams);
            if (numParams > 0)
                MPI_Bcast(task.appParams.data(), numParams, MPI_INT, 0, comm);

            // Reset runtime-only fields
            task.startTick = 0;
            task.duration  = 0;
        }
    }
}

// A POD struct safe to send with MPI
struct TaskLogRow {
    int step;
    int taskId;
    int coreId;
    unsigned long startTick;
    unsigned long duration;
    int nextTCore;
    int outgoingTrafficSize;
};

// Centralized CSV logging using TaskLogRow (POD-safe)
void logAllRanksTaskStatisticsCSV(const std::vector<Task> &localTasks,
                                  int rank, int step, int worldSize) {
    // Pack local tasks into POD rows
    std::vector<TaskLogRow> localRows;
    localRows.reserve(localTasks.size());
    for (const auto &t : localTasks) {
        TaskLogRow r{};
        r.step = step;
        r.taskId = t.taskId;
        r.coreId = t.coreId;
        r.startTick = t.startTick;
        r.duration = t.duration;
        r.nextTCore = t.nextTCore;
        r.outgoingTrafficSize = t.outgoingTrafficSize;
        localRows.push_back(r);
    }

    if (rank == 0) {
        // Start with rank 0's rows
        std::vector<TaskLogRow> allRows = localRows;

        // Receive rows from others
        for (int src = 1; src < worldSize; ++src) {
            int n = 0;
            MPI_Recv(&n, 1, MPI_INT, src, 1000 + step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (n > 0) {
                std::vector<TaskLogRow> buf(n);
                MPI_Recv(buf.data(), n * sizeof(TaskLogRow), MPI_BYTE,
                         src, 2000 + step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                allRows.insert(allRows.end(), buf.begin(), buf.end());
            }
        }

        // Write/append CSV
        const std::string filename = "task_log_all_ranks.csv";
        std::ofstream ofs;
        if (step == 0) {
            ofs.open(filename, std::ios::out);
            ofs << "Step,TaskID,CoreID,StartTick,Duration,NextTCore,OutgoingTrafficSize\n";
        } else {
            ofs.open(filename, std::ios::app);
        }
        for (const auto &r : allRows) {
            ofs << r.step << ','
                << r.taskId << ','
                << r.coreId << ','
                << r.startTick << ','
                << r.duration << ','
                << r.nextTCore << ','
                << r.outgoingTrafficSize << '\n';
        }
        ofs.close();
    } else {
        // Send count then packed rows
        int n = static_cast<int>(localRows.size());
        MPI_Send(&n, 1, MPI_INT, 0, 1000 + step, MPI_COMM_WORLD);
        if (n > 0) {
            MPI_Send(localRows.data(), n * sizeof(TaskLogRow), MPI_BYTE,
                     0, 2000 + step, MPI_COMM_WORLD);
        }
    }
}

