using Pkg
Pkg.develop(path="C:/nextCloud/MIB/SolidStateDetectors-Silicon.jl")  # z.B. "./SolidStateDetectors.jl"

using Unitful
using LinearAlgebra
using Plots
using Printf
using StaticArrays
using SolidStateDetectors
using SolidStateDetectors: SSDFloat, AbstractChargeDriftModel
using SolidStateDetectors: Electron, Hole, ChargeCarrier
T = Float32


### example from:
### https://github.com/JuliaPhysics/SolidStateDetectors.jl/pull/476
### 

using StatsBase
using StaticArrays

sim = Simulation{T}(SSD_examples[:TrueCoaxial])
det_z=T(10/1000) # m
det_r=T(10/1000) # m
sim.detector = SolidStateDetector(sim.detector, contact_id = 1, contact_potential = 0)
calculate_electric_potential!(sim)
calculate_electric_field!(sim)

# define the drifting function
# stop the drifting to demonstrate the diffusion 
# use constant zero for drifting, while using z-linear mu for diffusion
struct CustomChargeDriftModel{T <: SSDFloat} <: AbstractChargeDriftModel{T} 
    det_z::T
    calculate_mobility::Function
end
function SolidStateDetectors.getVe(fv::SVector{3, T}, cdm::CustomChargeDriftModel, current_pos::CartesianPoint{T})::SVector{3, T} where {T <: SSDFloat}
    mu0=0
    return -fv*mu0
end
function SolidStateDetectors.getVh(fv::SVector{3, T}, cdm::CustomChargeDriftModel, current_pos::CartesianPoint{T})::SVector{3, T} where {T <: SSDFloat}
    mu0=0
    return fv*mu0
end
function calculate_mobility(current_pos::CartesianPoint{T}, CC::Type{Hole}, det_z::T = det_z)
    current_pos[3]/det_z*4.2 # in m2/V/s
end
function calculate_mobility(current_pos::CartesianPoint{T}, CC::Type{Electron}, det_z::T = det_z)
    current_pos[3]/det_z*4.2 # in m2/V/s
end
charge_drift_model = CustomChargeDriftModel{T}(det_z, calculate_mobility)
sim.detector = SolidStateDetector(sim.detector, charge_drift_model)

sigma_list = []
startz_list = (0.1:0.1:0.9)*det_z
for startz in startz_list
    start= [det_r/2,0,T(startz)]
    nbcc = NBodyChargeCloud(CartesianPoint{T}(start), 100u"keV", 100, radius = T(0))
    evt = Event(nbcc)
    simulate!(evt, sim, diffusion = true, max_nsteps = 100)
    end_position_list=[]
    for path in evt.drift_paths
        push!(end_position_list, path.h_path[end])
    end
    end_position_list = permutedims(reduce(hcat, end_position_list))
    push!(sigma_list, std(end_position_list[:,1]))
end
p=plot(startz_list*1000, sigma_list*1000)
plot!(xlabel = "z / mm", legend=false, ylabel = "\$\\sigma\$ / mm", title="Group size (100 carriers), after 100 ns", size = (700,500))


###
### test for silicon
###

function ccs(evt,sim; diffusion = false, self_repulsion = false, nsteps = 500, self_repulsion_min_dist = T(1e-7))
    simulate!(evt, sim, diffusion = diffusion, self_repulsion = self_repulsion, max_nsteps = nsteps, self_repulsion_min_dist = self_repulsion_min_dist)
    ts = evt.drift_paths[1].timestamps_e * 1e6u"µs"
    Es = SolidStateDetectors.flatview(evt.energies[1])
    sig = [sqrt(sum(norm.(getindex.(getfield.(evt.drift_paths, :e_path), i) .- Ref(center)).^2 .* Es) / sum(Es)) for i in eachindex(evt.drift_paths[1].e_path)] * 1e3u"mm" 
    return ts, sig
end

# LsqFit curve_fit is not converging well, the following works better...
using LeastSquaresOptim
function curvefit(func,x,y,p0)
    # residuals for LastSquaresOptim
    function residuals!(res, p)
        res .= func(ustrip.(x), p) .- ustrip.(y)
    end
    lsq_problem = LeastSquaresProblem(x = copy(p0), f! = residuals!, output_length = length(y))
    fit_lsoptim = optimize!(lsq_problem, LevenbergMarquardt())
    return fit_lsoptim.minimizer
end

function Dif(t,p)
    return sqrt.(6*p[1] .*t)
end

path="test/benchmarks/SegBESi.yaml"
sim = Simulation{T}(path)
sim.detector = SolidStateDetector(sim.detector, contact_id = 1, contact_potential = 0);
calculate_electric_potential!(sim, refinement_limits = missing)
calculate_electric_field!(sim)

struct CustomChargeDriftModel{T <: SSDFloat} <: AbstractChargeDriftModel{T} 
    det_z::T
    calculate_mobility::Function
end
function SolidStateDetectors.getVe(fv::SVector{3, T}, cdm::CustomChargeDriftModel, current_pos::CartesianPoint{T})::SVector{3, T} where {T}
    mu0=0.002 # in m^2/V/s
    return -fv*mu0
end
function SolidStateDetectors.getVh(fv::SVector{3, T}, cdm::CustomChargeDriftModel, current_pos::CartesianPoint{T})::SVector{3, T} where {T}
    mu0=0.002 # in m^2/V/s
    return fv*mu0
end
function calculate_mobility(E::CartesianVector{T}, ::Type{CC}) where {T, CC <: ChargeCarrier}
    0.002 # in m^2/V/s
end
# Hamburg High Field Model : https://allpix-squared.docs.cern.ch/docs/06_models/02_carrier_mobility/
# function calculate_mobility(E::CartesianVector{T},
#                             Tcrystal::T,
#                             ::Type{CC}) where {T, CC <: ChargeCarrier}

#     # Feldbetrag in V/cm (Hamburg-Formeln sind in cm definiert)
#     E_cm = norm(E) / T(100)

#     # Temperatur-Skalierung
#     tscale = Tcrystal / T(300)

#     if CC == Electron
#         # ---- Elektronen ----
#         μ0_cm   = T(1430) * tscale^(-T(1.99))     # cm^2/(V*s)
#         v_sat   = T(1.05e7) * tscale^(-T(0.302))  # cm/s

#         invμ = (one(T)/μ0_cm) + (E_cm / v_sat)

#     elseif CC == Hole
#         # ---- Löcher ----
#         μ0_cm   = T(457) * tscale^(-T(2.80))      # cm^2/(V*s)
#         b       = T(9.57e-8) * tscale^(-T(0.155)) # s/cm
#         c       = T(-3.24e-13)                    # s/V  (kein T-scaling)
#         E0      = T(2970) * tscale^(T(0.563))     # V/cm

#         if E_cm < E0
#             invμ = one(T)/μ0_cm
#         else
#             dE   = E_cm - E0
#             invμ = (one(T)/μ0_cm) + b*dE + c*dE^2
#         end
#     else
#         error("Unbekannter ChargeCarrier: $CC")
#     end

#     # Umwandlung cm^2/(V*s) → m^2/(V*s)  (× 1e-4)
#     μ_m = (one(T)/invμ) * T(1e-4)
#     return μ_m
# end
charge_drift_model = CustomChargeDriftModel{T}(42, calculate_mobility) # first variable not used
sim.detector = SolidStateDetector(sim.detector, charge_drift_model)

center = CartesianPoint{T}(0,0,0.02)
E = 1460u"keV"
#nbcc = NBodyChargeCloud(CartesianPoint{T}(0,0,0), 6*u"keV", 100, radius = T(3e-5), number_of_shells = 4)

nbcc = NBodyChargeCloud(center, E, 1000, radius = 0.0001f0)
evt = Event(nbcc);

dts,dsig = ccs(evt,sim; diffusion=true)
plot(dts,dsig, label="diffusion", xlabel="t", ylabel="sigma")

dfit = curvefit(Dif, dts, dsig, [0.01])
plot!(dts,Dif(ustrip.(dts),dfit)*u"mm", label=@sprintf("D = %.1f cm²/s", dfit[1]*1e4))

rts,rsig = ccs(evt,sim; self_repulsion=true, self_repulsion_min_dist = T(1e-7))
plot!(rts,rsig, label="repulsion")

rfit = curvefit(Rep, rts, rsig, [0.2])
plot!(rts,Rep(ustrip.(rts),rfit)*u"mm", label=@sprintf("C = %.1f cm³/s", rfit[1]*1e3))

drts,drsig = ccs(evt,sim; diffusion=true, self_repulsion=true, self_repulsion_min_dist = 1e-7)
plot!(drts,drsig, label="dif + rep")

drfit = curvefit(rykov,drts,drsig,[0.15,0.3,0.03])
plot!(drts, rykov(ustrip.(drts), drfit), label=@sprintf("D = %.1f cm²/s, C = %.1f cm³/s", drfit[3]*1e4, drfit[2]*1e3))

### gifs

function plot_step_j(event, j = Int(1e9), i = 1, lims = (-0.0035,0.0035))
    plot(xlabel = "x/m", ylabel = "y/m", zlabel = "z/m", size = (500,500))
    vertex = 0
    for (k,c) in enumerate([1,nbcc.shell_structure...])
        for dp in event.drift_paths[vertex+1:vertex+c]
            plot!(dp.h_path[max(1,i):min(j,end)], label = "", lw = 1, color = :red)
            scatter!(dp.h_path[min(j,end)], label = "", markersize = 10 * exp(-(k-1)), color = :red, markerstrokewidth = 0)
        end
        vertex += c
    end
    plot!(xlims = lims, ylims = lims, zlims = lims .+ 0.02,
    xticks = (-0.003:0.0015:0.003,-0.003:0.0015:0.003), yticks = (-0.003:0.0015:0.003,-0.003:0.0015:0.003), zticks = (0.017:0.0015:0.023, -0.003:0.0015:0.003))
end

# Diffusion
simulate!(evt, sim, diffusion = true, max_nsteps = 50000)
diffusion = @animate for j in 1:2000:50000 plot_step_j(evt, j, j - 2000) end
gif(diffusion, fps=10)

# Self-repulsion
simulate!(evt, sim, self_repulsion = true, max_nsteps = 20000)
repulsion = @animate for j in 1:1000:20000 plot_step_j(evt, j) end
gif(repulsion, fps=10)

# Diffusion and self-repulsion
simulate!(evt, sim, self_repulsion = true, diffusion = true, max_nsteps = 20000)
both = @animate for j in 1:1000:20000 plot_step_j(evt, j, j - 2000) end
gif(both, fps=10)



