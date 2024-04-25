module Analysis

using Plots
using Statistics
using DataFrames

"""
    plot_results(times, memory, errors)

Plot the results of a Cholesky decomposition analysis.

# Arguments
- `times::Array{Float64,1}`: Array of time measurements for each iteration.
- `memory::Array{Int64,1}`: Array of memory usage measurements for each iteration.
- `errors::Array{Float64,1}`: Array of error measurements for each iteration.

# Returns
- `nothing`

"""
function plot_results(times, memory, errors)
    p1 = plot(times, label="Time", title="Cholesky Decomposition", xlabel="Iteration",
        ylabel="Time (s)", linewidth=2, legend=:topleft)
    p2 = plot(memory, label="Memory", title="Cholesky Decomposition", xlabel="Iteration", ylabel="Memory (bytes)")
    p3 = plot(errors, label="Error", title="Cholesky Decomposition", xlabel="Iteration", ylabel="Error")

    plot(p1, p2, p3)
end

"""
    ShowStats(times::Array{Float64,1}, memory::Array{Int64,1}, errors::Array{Float64,1})

Prints the statistics of the time, memory, and error measurements.

# Arguments
- `times::Array{Float64,1}`: Array of time measurements for each iteration.
- `memory::Array{Int64,1}`: Array of memory usage measurements for each iteration.
- `errors::Array{Float64,1}`: Array of error measurements for each iteration.

# Returns
- `nothing`

"""
function ShowStats(times::Array{Float64,1}, memory::Array{Int64,1}, errors::Array{Float64,1})
    stats_times = [mean(times), median(times), std(times), minimum(times), maximum(times)]
    stats_memory = [mean(memory), median(memory), std(memory), minimum(memory), maximum(memory)]
    stats_errors = [mean(errors), median(errors), std(errors), minimum(errors), maximum(errors)]


    df = DataFrame(Vettore=["Times", "Memory", "Errors"],
        Media=[stats_times[1], stats_memory[1], stats_errors[1]],
        Mediana=[stats_times[2], stats_memory[2], stats_errors[2]],
        Deviazione_Standard=[stats_times[3], stats_memory[3], stats_errors[3]],
        Minimo=[stats_times[4], stats_memory[4], stats_errors[4]],
        Massimo=[stats_times[5], stats_memory[5], stats_errors[5]])

    println(df)
end

"""
    Visualizations(times::Array{Float64,1}, memory::Array{Int64,1}, errors::Array{Float64,1})

Generates visualizations of the results of a Cholesky decomposition analysis.

# Arguments
- `times::Array{Float64,1}`: Array of time measurements for each iteration.
- `memory::Array{Int64,1}`: Array of memory usage measurements for each iteration.
- `errors::Array{Float64,1}`: Array of error measurements for each iteration.

# Returns
- `nothing`
"""
function Visualizations(times::Array{Float64,1}, memory::Array{Int64,1}, errors::Array{Float64,1})
    ShowStats(times, memory, errors)
    plot_results(times, memory, errors)

end

end