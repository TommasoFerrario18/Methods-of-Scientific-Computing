module Analysis

using Plots
using Statistics
using DataFrames

"""
    plot_results(times, memory, errors)

Plot the results of a Cholesky decomposition analysis.

# Arguments
- `times::Vector{Float64}`: Array of time measurements for each iteration.
- `memory::Vector{Float64}`: Array of memory usage measurements for each iteration.
- `errors::Vector{Float64}`: Array of error measurements for each iteration.

# Returns
- `nothing`

"""
function plot_results(times::Vector{Float64}, memory::Vector{Float64}, errors::Vector{Float64}, title::String)
    p1 = plot(times,  label="Time",   title=title, xlabel="Iteration", ylabel="Time (s)")
    p2 = plot(memory, label="Memory", title=title, xlabel="Iteration", ylabel="Memory (MB)")
    p3 = plot(errors, label="Error",  title=title, xlabel="Iteration", ylabel="Error")
    plot(p1, p2, p3, layout=(3,1), size=(800, 800))
end

"""
    ShowStats(times::Vector{Float64}, memory::Vector{Float64}, errors::Vector{Float64})

Prints the statistics of the time, memory, and error measurements.

# Arguments
- `times::Vector{Float64}`: Array of time measurements for each iteration.
- `memory::Vector{Float64}`: Array of memory usage measurements for each iteration.
- `errors::Vector{Float64}`: Array of error measurements for each iteration.

# Returns
- `nothing`

"""
function ShowStats(times::Vector{Float64}, memory::Vector{Float64}, errors::Vector{Float64})
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
    Visualizations(times::Vector{Float64}, memory::Vector{Float64}, errors::Vector{Float64})

Generates visualizations of the time, memory, and error measurements.

# Arguments
- `times::Vector{Float64}`: Array of time measurements for each iteration.
- `memory::Vector{Float64}`: Array of memory usage measurements for each iteration.
- `errors::Vector{Float64}`: Array of error measurements for each iteration.

# Returns
- `nothing`

"""
function Visualizations(times::Vector{Float64}, memory::Vector{Float64}, errors::Vector{Float64}, title::String)
    ShowStats(times, memory, errors)
    plot_results(times, memory, errors, title)

end

end