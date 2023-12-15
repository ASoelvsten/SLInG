using Catalyst, DifferentialEquations, IfElse
hillpn(p,a,K,n) = IfElse.ifelse(a>0.0, hill(p, a, K, n), hillr(p, abs(a), K, n))

rn = @reaction_network begin
    hillpn(p2, a1 * k11, k6, k5), ∅ --> m1
    hillpn(p3, a2 * k11, k6, k5), ∅ --> m1
    hillpn(p4, a3 * k11, k6, k5), ∅ --> m1
    hillpn(p1, a4 * k12, k6, k5), ∅ --> m2
    hillpn(p3, a5 * k12, k6, k5), ∅ --> m2
    hillpn(p4, a6 * k12, k6, k5), ∅ --> m2
    hillpn(p1, a7 * k13, k6, k5), ∅ --> m3
    hillpn(p2, a8 * k13, k6, k5), ∅ --> m3
    hillpn(p4, a9 * k13, k6, k5), ∅ --> m3
    hillpn(p1, a10 * k14, k6, k5), ∅ --> m4
    hillpn(p2, a11 * k14, k6, k5), ∅ --> m4
    hillpn(p3, a12 * k14, k6, k5), ∅ --> m4
    k11, ∅ --> m1
    k2 + growthr, m1 --> ∅
    k3, m1 --> m1 + p1
    k4 + growthr, p1 --> ∅
    k12, ∅ --> m2
    k2 + growthr, m2 --> ∅
    k3, m2 --> m2 + p2
    k4 + growthr, p2 --> ∅
    k13, ∅ --> m3
    k2 + growthr, m3 --> ∅
    k3, m3 --> m3 + p3
    k4 + growthr, p3 --> ∅
    k14, ∅ --> m4
    k2 + growthr, m4 --> ∅
    k3, m4 --> m4 + p4
    k4 + growthr, p4 --> ∅
end
u0 = [:m1 => ceil.(Int, (0.36787944117207516 / (0.04987442692380366 + 0.03))), :m2 => ceil.(Int, (0.36787944117207516 / (0.04987442692380366 + 0.03))), :m3 => ceil.(Int, (0.36787944117207516 / (0.04987442692380366 + 0.03))), :m4 => ceil.(Int, (0.36787944117207516 / (0.04987442692380366 + 0.03))), :p1 => ceil.(Int, ((0.36787944117207516 * 4.3376521035785345) / ((0.04987442692380366 + 0.03) * (0.36777724535966944 + 0.03)))), :p2 => ceil.(Int, ((0.36787944117207516 * 4.3376521035785345) / ((0.04987442692380366 + 0.03) * (0.36777724535966944 + 0.03)))), :p3 => ceil.(Int, ((0.36787944117207516 * 4.3376521035785345) / ((0.04987442692380366 + 0.03) * (0.36777724535966944 + 0.03)))), :p4 => ceil.(Int, ((0.36787944117207516 * 4.3376521035785345) / ((0.04987442692380366 + 0.03) * (0.36777724535966944 + 0.03))))]
tspan = (0.0, 50.0)
p = [:k11 => 0.36787944117207516, :k12 => 0.36787944117207516, :k13 => 0.36787944117207516, :k14 => 0.36787944117207516, :k2 => 0.04987442692380366, :k3 => 4.3376521035785345, :k4 => 0.36777724535966944, :k5 => 10.0, :k6 => 61.47011276535837, :growthr => 0.03, :a1 => 0.0, :a2 => 0.0, :a3 => 3.0, :a4 => 0.0, :a5 => 0.0, :a6 => 0.0, :a7 => 0.0, :a8 => 0.0, :a9 => 0.0, :a10 => 0.0, :a11 => 0.0, :a12 => 0.0];
dprob = DiscreteProblem(rn, u0, tspan, p)
jprob = JumpProblem(rn, dprob, Direct())
prob = EnsembleProblem(jprob)

function ssa_sampler(k11, k12, k13, k14, k2, k3, k4, k5, k6, growthr, cell_num, a)
    p = [a[1], k11, k6, k5, a[2], a[3], a[4], k12, a[5], a[6], a[7], k13, a[8], a[9], a[10], k14, a[11], a[12], k2, growthr, k3, k4]

    u0 = float.([ceil.(Int, (k11 / (k2 + growthr))), ceil.(Int, (k12 / (k2 + growthr))), ceil.(Int, (k13 / (k2 + growthr))), ceil.(Int, (k14 / (k2 + growthr))), ceil.(Int, ((k11 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k12 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k13 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k14 * k3) / ((k2 + growthr) * (k4 + growthr))))])

    prob2 = remake(prob; u0=u0, p=p)
    esol = solve(prob2, SSAStepper(), trajectories=cell_num)

    mRNA1 = [esol[i].u[end][1] for i in 1:cell_num]
    mRNA2 = [esol[i].u[end][2] for i in 1:cell_num]
    mRNA3 = [esol[i].u[end][3] for i in 1:cell_num]
    mRNA4 = [esol[i].u[end][4] for i in 1:cell_num]
    ground_truth_data = hcat(mRNA1, mRNA2, mRNA3, mRNA4)
    return ground_truth_data
end

sprob = SDEProblem(rn, u0, tspan, p)
prob2 = EnsembleProblem(sprob)

function sde_sampler(k11, k12, k13, k14, k2, k3, k4, k5, k6, growthr, cell_num, a)
    p = [a[1], k11, k6, k5, a[2], a[3], a[4], k12, a[5], a[6], a[7], k13, a[8], a[9], a[10], k14, a[11], a[12], k2, growthr, k3, k4]
    u0 = float.([ceil.(Int, (k11 / (k2 + growthr))), ceil.(Int, (k12 / (k2 + growthr))), ceil.(Int, (k13 / (k2 + growthr))), ceil.(Int, (k14 / (k2 + growthr))), ceil.(Int, ((k11 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k12 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k13 * k3) / ((k2 + growthr) * (k4 + growthr)))), ceil.(Int, ((k14 * k3) / ((k2 + growthr) * (k4 + growthr))))])
    prob3 = remake(prob2; u0=u0, p=p)
    esol2 = try
        solve(prob3, EulerHeun(), dt=0.1, trajectories=cell_num)
    catch
        false
    end

    if esol2 == false
        ground_truth_data = zeros(500, 4)
    else
        mRNA1 = [esol2[i].u[end][1] for i in 1:cell_num]
        mRNA2 = [esol2[i].u[end][2] for i in 1:cell_num]
        mRNA3 = [esol2[i].u[end][3] for i in 1:cell_num]
        mRNA4 = [esol2[i].u[end][4] for i in 1:cell_num]
        ground_truth_data = hcat(mRNA1, mRNA2, mRNA3, mRNA4)
    end
    return ground_truth_data
end

