#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

RuntimeEnvironment::RuntimeEnvironment() {}
RuntimeEnvironment::~RuntimeEnvironment() {}
bool RuntimeEnvironment::is_master_node(EnvironmentRole role) {
    return rank_id(role) == 0;
}
std::string format_timestamp(time_t time, const char* format) {
    std::string result;
    struct tm p = *localtime(&time);
    char time_str_buffer[64];
    int size = strftime (time_str_buffer, 64, format, &p);
    if (size > 0) {
        result.assign(time_str_buffer, size);
    }
    return result;
}

struct MpiNodeInfo {
    int rank_id = -1;
    int node_num = 0;
    MPI_Comm mpi_comm;
};

class MPIRuntimeEnvironment : public RuntimeEnvironment {
public:
    MPIRuntimeEnvironment() {}
    virtual ~MPIRuntimeEnvironment() {}
    virtual int initialize(YAML::Node config) {
        return 0;
    }
    virtual int wireup() {
        int hr = MPI_Init(NULL, NULL);
        if (MPI_SUCCESS != hr) {
            LOG(FATAL) << "MPI_init failed with error code" << hr; 
            return -1;
        }
        _roles_node_info.resize(static_cast<int>(EnvironmentRole::ALL) + 1);
        set_role(EnvironmentRole::ALL);
        return 0;
    }

    virtual uint32_t rank_id(EnvironmentRole role) {
        return mpi_node_info(role).rank_id;
    }
    virtual uint32_t node_num(EnvironmentRole role) {
        return mpi_node_info(role).node_num;
    }
    virtual int set_role(EnvironmentRole role) {
        auto& node_info = mpi_node_info(role);
        if (node_info.rank_id < 0) {
            if (role == EnvironmentRole::ALL) {
                node_info.mpi_comm = MPI_COMM_WORLD;
            } else {
                MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(role), 
                    mpi_node_info(EnvironmentRole::ALL).rank_id, &(node_info.mpi_comm));
            }
            MPI_Comm_rank(node_info.mpi_comm, &(node_info.rank_id));
            MPI_Comm_size(node_info.mpi_comm, &(node_info.node_num));
        }
        return 0;
    }

    virtual void barrier(EnvironmentRole role) {
        MPI_Barrier(mpi_node_info(role).mpi_comm);
    }
    virtual void bcast(paddle::framework::BinaryArchive& ar, int root_id, EnvironmentRole role) {
        auto& node_info = mpi_node_info(role);
        int len = (int)ar.length();
        MPI_Bcast(&len, 1, MPI_INT, root_id, node_info.mpi_comm);
        ar.resize(len);
        ar.set_cursor(ar.buffer());
        MPI_Bcast(ar.buffer(), len, MPI_BYTE, root, node_info.mpi_comm);
    }
protected:
    virtual void print_log(EnvironmentLogType type, EnvironmentLogLevel level,  const std::string& log_str);
    inline MpiNodeInfo& mpi_node_info(EnvironmentRole role) {
        return _roles_node_info[static_cast<int>(role)];
    }
private:
    std::vector<MpiNodeInfo> _roles_node_info;
    
};

REGISTER_CLASS(RuntimeEnvironment, MPIRuntimeEnvironment);
    

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
