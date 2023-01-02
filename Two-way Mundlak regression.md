$d_i$: Are you eventually treated?
$p_t$: Are we in a post-treatment period?
$w_i = d_i \cdot p_t$
$$y_{it} = \alpha + \beta w_i + d_i + p_t + u_{it}  $$
> You are not doing anything fancy. [[https://www.dropbox.com/sh/zj91darudf2fica/AADj_jaf5ZuS1muobgsnxS6Za?dl=0&preview=did_seminar_20210923.mp4| Wooldridge in DiD seminar 23.09.21]]

The point here is that Wooldridge shows that the TWFE estimator is equivalent to this "pooled" regression which is exactly the difference-in-differences estimator (difference before and after treatment for the treated group versus before after for the control group). 

==So what happens when we have heterogeneous effects in a staggered treatment setting?==

> Is is not a problem with our estimator. It is a problem with our imagination. (Wooldridge)

Wooldridge notes that $w_i \cdot fr_t = d_i \cdot fr_t$ : Interacting the treatment indicator with the time-dummies is the same as interacting the time-constant $d_i$ with the time-dummies $fr_t$.    The model he comes up with for correct parameterization of the case with staggered treatments and heterogeneous treatment effects is the following:
$$Y_{it} = \alpha + \beta_q (w_{it} \cdot fq_t) + ... + \beta_T (w_{it} \cdot fT_t) + \xi d_i + \theta_q fq_t + ... + \theta_T fT_t + u_{it},$$
$$t = 1, ..., T; i = 1,2,...,N$$
$fq_t ... fT_t$  denote the fact that we are in a staggered treatment setting and that different cohorts have different pre- and post-treatment periods. $q$ is the first period someone enters treatment. We could add dummies for time $t < q$, but these are redundant.

In the above equation, we allow treatment effects $\beta_{q ... T}$ to change over treatment period. Below, we show how we can simply specify this model in R (including several redundant parameters).

```r
library(tidyverse)
library(fixest)
library(did2s)
library(did)

set.seed(100)

df <- did2s::gen_data(
  g1 = 2011, # treatment date for group 1
  g2 = 2012, # treatment date for group 2
  g3 = 0, # treatment date for group 3
  panel = c(2010, 2012), # start and end years for panel
  te1 = 3, # treatment effect for group 1
  te2 = 1, # treatment effect for group 2
  te3 = 0, #  treatment effect for group 3
  te_m1 = 0, # treatment effect slope per year for group 1
  te_m2 = 0, # treatment effect slope per year for group 2
  te_m3 = 0, # treatment effect slope per year for group 3
  n = 1500 # number of individual in sample
)

df_hom <- did2s::gen_data(panel = c(2010, 2012),
                          g1 = 2011, g2 = 2012, g3 = 0,
                          te1 = 2, te2 = 2, te3 = 0,
                          te_m1 = 0, te_m2 = 0, te_m3 = 0)
# df is a simulated dataset with heterogeneous and staggered treatment effects (the group treatd in 2011 has effect = 3, while the group treated one year later has effect = 1). A first-mover advantage is common in many real applications.
# df_hom is a simulated dataset with homogeneous and staggered treatment effects. Here, the expected (average) effect in both treatment cohorts are the same.

# Two-Way Fixed-Effects with homogeneous treatment effects
feols(dep_var ~ treat | unit + year, data = df_hom, cluster = "state") |> summary()
# Two-Way Mundlak with homogeneous treatment effects
df_hom <- df_hom |> 
  group_by(unit) |> mutate(mtreat_u = mean(treat)) |> 
  group_by(year) |> mutate(mtreat_t = mean(treat))
lm(dep_var ~ treat + mtreat_u + mtreat_t, data = df_hom) |> summary()

# This follows an equivalent logic, note that year dummies t < q are redundant
feols(dep_var ~ treat | year + g, data = df_hom) |> summary()

# So far, everything is good (although almost by coincidence, as we were lucky that the effect for the two cohorts were exactly the same!)

# The naive Two-Way Mundlak and TWFE is wrongly parameterized when we have heterogeneous TEs.
# This produces an average effect, but not 3 for group 1 and 1 for group 2.
df <- df |> 
  group_by(unit) |> mutate(mtreat_u = mean(treat)) |> 
  group_by(year) |> mutate(mtreat_t = mean(treat))
lm(dep_var ~ treat + mtreat_u + mtreat_t, data = df) |> summary()

# Similarly wrong
feols(dep_var ~ treat | year + g, data = df) |> summary()

# Instead, we can add interactions between treatment cohort (g) and time-since treatment (rel_year)
# Note that more than one level for rel_year t < q is redundant, and we could have collapsed these.
# This we could have done for year t < q too if we wanted.
# Note that we need to provide correct base-line comparisons to get correctly formed estimates.
levels(factor(df$rel_year))
feols(dep_var ~  factor(g):factor(rel_year) | year , data = df) |> summary()
# Adding the treatment indicator is also redundant
feols(dep_var ~ treat : factor(g) : factor(rel_year) | year , data = df) |> summary()

# Estimates for groups, calendar time, or "event study" (grouped by rel_year) can be aggregated from here
# An easier approach is to use specialized functions such as those in the did-package
did::att_gt(yname = "dep_var", gname = "g", idname = "unit", tname = "year", 
           control_group = "notyettreated", data = df) |> 
  did::aggte(type = "group") 

# control_group could be restricted to "nevertreated"  
# type can also be "dynamic" (event study),  "calendar", or "simple" (ATT)

# Compare the "simple" approach with the wrongly parameterized model to see the problem of using the naive TWFE/TWM to estimate the ATT

```

One nice thing about the TWM is that it allows time-constant covariates $X_i$. These are particularly useful if we include them as interactions with time dummies and treatment. $X_i$ would then become a mediator. Doing this would be a way to explore treatment heterogeneity across observed covariates. It should be mentioned here that the differences in estimated effects (e.g., across cohorts or across $X_i$) are not causal estimates themselves. We would then need to consider the selection mechanism into cohorts or into different $X_i$.

We have also not discussed the assumptions needed to interpret the estimate of the treatment as causal here, such as SUTVA, the parallel trends assumption, and the no anticipation assumption. In most observable settings, some or all of these assumptions are violated (to some degree). It is therefore always advisable to test the model in several ways, e.g., through a discussion of how the model relates to other things we belive is true, through testing the predictive capacity of the model, through building alternative explanations, etc. Furthermore, even if the causal estimate is correct, there are no guarantees that causal effects in the social context would stay the same over time or for other cases. Exploring treatment heterogeneity is also a way to build some intuition for how effects might change in new settings.