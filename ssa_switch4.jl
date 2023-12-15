function ssa_switch4(temp_data)

    c = zeros(temp_data.M)
    for k = 1:temp_data.M
        c[k] = rates(temp_data, k)
    end
    c_cum = cumsum(c, dims=1)
    c0 = c_cum[end]
    t = 0
    while t < temp_data.tau
        # find which reaction will fire
        drxn = rand() * c0
        psm = 0.0
        reaction = 0
        while psm < drxn
            reaction = reaction + 1
            psm = psm + c[reaction]
        end
        # compute waiting time
        dt = -log(rand()) / c0

        # update species after reaction
        for k = 1:4
            ind = temp_data.cm_rea[reaction, k]
            if !iszero(temp_data.cm_rea[reaction, k])
                temp_data.X[ind] = temp_data.X[ind] + temp_data.stoichio[reaction, k]
            end
        end

        # update rates involving species of 'reaction' after species update
        c = update_rates(c, temp_data, reaction)
        c_cum = cumsum(c, dims=1)
        c0 = c_cum[end]

        t = t + dt
    end
    return temp_data.X
end

