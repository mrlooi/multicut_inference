#pragma once

#include <andres/marray.hxx>
#include <andres/graph/graph.hxx>

#include <nl-lmp/solve-joint.hxx>



void enforce_constraints(nl_lmp::Problem<andres::graph::Graph<>>& problem, size_t number_of_bg_classes)
{
	auto const numberOfClasses = problem.numberOfClasses();

	for (size_t e = 0; e < problem.liftedGraph().numberOfEdges(); ++e)
    {
        auto const v0 = problem.liftedGraph().vertexOfEdge(e, 0);
        auto const v1 = problem.liftedGraph().vertexOfEdge(e, 1);

        for (size_t k = 0; k < numberOfClasses; ++k)
            for (size_t l = 0; l < numberOfClasses; ++l)
                if (k != l)
                    // set very high cost for joining vertices with different labels
                    problem.setPairwiseJoinCost(v0, v1, k, l, 1000.0, e);
                else
                    problem.setPairwiseJoinCost(v0, v1, k, l, -problem.getPairwiseCutCost(v0, v1, k, l, e), e);

        // set high cost for cutting vertices of the background classes
        // i.e. classes that don't have instances in CityScapes
        for (size_t k = 0; k < number_of_bg_classes; ++k)
            problem.setPairwiseCutCost(v0, v1, k, k, 1000.0, e);
    }
}

// 
// 'unaries' is a table which rows are spanned by vertices and columns -- by classes.
//
// 'general_pw' is a table each row of which has the format: v0 v1 cost
// it assigns the same cut cost to any combintation of classes on the vertices
// 
// 'specific_pw' is a table each row of which has the format: v0 v1 c0 c1 cost
// it assigns a particular cost for two vertices v0 and v1 labeled with c0 and c1 correspondingly
// 
// 'number_of_bg_classes' -- CityScapes has some "backgorund" classes that don't have instance annotations, so we force them not to have boundaries (see above)
// 
// 'sol' is the output classification and clustering
// 
void solve(andres::View<double> const& unaries, andres::View<double> const& general_pw, andres::View<double> const& specific_pw, size_t number_of_bg_classes, andres::View<int>& sol)
{
    auto const numberOfVertices = unaries.shape(0);
    auto const numberOfClasses = unaries.shape(1);

    nl_lmp::Problem<andres::graph::Graph<>> problem(numberOfVertices, numberOfClasses);

    for (size_t v = 0; v < numberOfVertices; ++v)
        for (size_t c = 0; c < numberOfClasses; ++c)
            problem.setUnaryCost(v, c, unaries(v, c));

    for (size_t e = 0; e < general_pw.shape(0); ++e)
        for (size_t k = 0; k < numberOfClasses; ++k)
            for (size_t l = 0; l < numberOfClasses; ++l)
                problem.setPairwiseCutCost(general_pw(e, 0) + 0.1, general_pw(e, 1) + 0.1, k, l, general_pw(e, 2));

    for (size_t e = 0; e < specific_pw.shape(0); ++e)
        problem.setPairwiseCutCost(specific_pw(e, 0) + 0.1, specific_pw(e, 1) + 0.1, specific_pw(e, 2) + 0.1, specific_pw(e, 3) + 0.1, specific_pw(e, 4));

	enforce_constraints(problem, number_of_bg_classes);

	nl_lmp::Solution solution(problem.numberOfVertices());

	// initialize vertex labels with the most probable class
    for (size_t v = 0; v < problem.numberOfVertices(); ++v)
    {
        double best_gain = problem.getUnaryCost(v, 0);
        size_t best_color = 0;

        for (size_t k = 1; k < problem.numberOfClasses(); ++k)
            if (problem.getUnaryCost(v, k) < best_gain)
            {
                best_gain = problem.getUnaryCost(v, k);
                best_color = k;
            }

        solution[v].classIndex = best_color;
    }

    // create initial multicut based on labels
    // normally, we would use GAEC to initialize the multicut solver, but here we have contraints that vertices of different classes should be in different components
    // one can still combine the result of the code below with GAEC -- we jsut didn't try it
    std::vector<char> visited(problem.numberOfVertices());
    std::stack<size_t> S;

    for (size_t i = 0, clusterIndex = 0; i < visited.size(); ++i)
        if (!visited[i])
        {
            visited[i] = 1;
            S.push(i);

            solution[i].clusterIndex = clusterIndex;

            auto classIndex = solution[i].classIndex;

            while (!S.empty())
            {
                auto v = S.top();
                S.pop();

                for (auto w = problem.originalGraph().verticesFromVertexBegin(v); w != problem.originalGraph().verticesFromVertexEnd(v); ++w)
                    if (!visited[*w] && solution[*w].classIndex == classIndex)
                    {
                        visited[*w] = 1;
                        solution[*w].clusterIndex = clusterIndex;
                        S.push(*w);
                    }
            }

            ++clusterIndex;
        }

    //  the solver itself
	solution = nl_lmp::update_labels_and_multicut(problem, solution);
    
    for (size_t v = 0; v < solution.size(); ++v)
    {
        sol(v, 0) = solution[v].classIndex;
        sol(v, 1) = solution[v].clusterIndex;
    }
}
