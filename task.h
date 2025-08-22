struct Task {
    int taskId;
    int coreId;
    int nextTCore;
    int outgoingTrafficSize;
    int appId;
    std::vector<int> appParams;

    unsigned long startTick = 0;   // will be assigned in main()
    unsigned long duration = 0;    // assigned after execution
};
