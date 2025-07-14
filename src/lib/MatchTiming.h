/************************************************************************
 *
 * (c) 2025 Machines With Vision Ltd.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Machines With Vision Limited and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Machines With Vision Limited
 * and its suppliers and may be covered by U.K. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Machines With Vision Limited.
 *
 ************************************************************************/

#ifndef MATCHTIMING_H
#define MATCHTIMING_H

namespace mwvcv
{
    struct MatchTiming
    {
        MatchTiming()
        {
            upload   = 0.0;
            match    = 0.0;
            download = 0.0;
        }

        double upload;   ///< Upload time from CPU to GPU in ms
        double match;    ///< Matching operation time in ms
        double download; ///< Download time from GPU to CPU in ms
    };

} // namespace mwvcv

#endif //MATCHTIMING_H
