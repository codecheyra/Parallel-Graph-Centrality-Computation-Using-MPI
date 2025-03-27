#include "template.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>

using namespace std;

void init_mpi(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
}

void end_mpi()
{
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int, int>> &partial_edge_list,
                               map<int, int> &partial_vertex_color, int k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> local_colors;
    local_colors.reserve(partial_vertex_color.size() * 2);
    for (auto &p : partial_vertex_color)
    {
        local_colors.push_back(p.first);
        local_colors.push_back(p.second);
    }
    int local_colors_size = local_colors.size();

    vector<int> all_colors_sizes(size);
    MPI_Allgather(&local_colors_size, 1, MPI_INT, all_colors_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    int total_colors = 0;
    for (int i = 0; i < size; i++)
    {
        displs[i] = total_colors;
        total_colors += all_colors_sizes[i];
    }

    vector<int> global_colors(total_colors);
    MPI_Allgatherv(local_colors.data(), local_colors_size, MPI_INT,
                   global_colors.data(), all_colors_sizes.data(), displs.data(), MPI_INT,
                   MPI_COMM_WORLD);

    unordered_map<int, int> global_vertex_color;
    for (size_t i = 0; i < global_colors.size(); i += 2)
    {
        int vertex = global_colors[i];
        int col = global_colors[i + 1];
        global_vertex_color[vertex] = col;
    }

    unordered_map<int, unordered_map<int, int>> local_centrality;

    local_centrality.reserve(partial_edge_list.size());
    for (auto &edge : partial_edge_list)
    {
        int u = edge.first;
        int v = edge.second;

        auto it_v = global_vertex_color.find(v);
        if (it_v != global_vertex_color.end())
        {
            local_centrality[u][it_v->second]++;
        }

        auto it_u = global_vertex_color.find(u);
        if (it_u != global_vertex_color.end())
        {
            local_centrality[v][it_u->second]++;
        }
    }

    for (auto &p : global_vertex_color)
    {
        int vertex = p.first;
        if (local_centrality.find(vertex) == local_centrality.end())
            local_centrality[vertex] = unordered_map<int, int>();
    }

    vector<int> local_data;
    local_data.reserve(local_centrality.size() * 3);
    for (auto &entry : local_centrality)
    {
        int vertex = entry.first;
        for (auto &inner : entry.second)
        {
            local_data.push_back(vertex);
            local_data.push_back(inner.first);
            local_data.push_back(inner.second);
        }
    }

    int local_size = local_data.size();
    vector<int> recvcounts;
    if (rank == 0)
        recvcounts.resize(size);
    MPI_Gather(&local_size, 1, MPI_INT,
               recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> global_data;
    if (rank == 0)
    {
        vector<int> displs_data(size, 0);
        int total_size = 0;
        for (int i = 0; i < size; i++)
        {
            displs_data[i] = total_size;
            total_size += recvcounts[i];
        }
        global_data.resize(total_size);
        MPI_Gatherv(local_data.data(), local_size, MPI_INT,
                    global_data.data(), recvcounts.data(), displs_data.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(local_data.data(), local_size, MPI_INT,
                    nullptr, nullptr, nullptr, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {

        unordered_map<int, unordered_map<int, int>> global_centrality;
        for (size_t i = 0; i < global_data.size(); i += 3)
        {
            int vertex = global_data[i];
            int col = global_data[i + 1];
            int cnt = global_data[i + 2];
            global_centrality[vertex][col] += cnt;
        }

        set<int> distinct_colors;
        for (auto &p : global_vertex_color)
            distinct_colors.insert(p.second);

        vector<vector<int>> result;
        for (int color : distinct_colors)
        {
            vector<pair<int, int>> vec;
            vec.reserve(global_vertex_color.size());
            for (auto &p : global_vertex_color)
            {
                int vertex = p.first;
                int count = 0;
                if (global_centrality.count(vertex) &&
                    global_centrality[vertex].count(color))
                    count = global_centrality[vertex][color];
                vec.push_back({vertex, count});
            }
            sort(vec.begin(), vec.end(), [](const pair<int, int> &a, const pair<int, int> &b)
                 {
                     if (a.second == b.second)
                         return a.first < b.first;
                     return a.second > b.second; });
            vector<int> topk;
            int limit = min(k, (int)vec.size());
            topk.reserve(limit);
            for (int i = 0; i < limit; i++)
                topk.push_back(vec[i].first);
            result.push_back(topk);
        }
        return result;
    }
    else
    {
        return vector<vector<int>>();
    }
}
